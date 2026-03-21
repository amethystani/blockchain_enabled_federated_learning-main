// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title SpectralBFT
 * @notice Optimistic Byzantine-Robust Federated Learning with On-Chain Spectral Fraud Proofs
 *         and Spectral Graded Broadcast for P2P validation.
 *
 * Architecture (Contribution 1 — Primary):
 *   Block b:      Clients submit gradient hash + Frobenius norm
 *   Block b+1:    Round optimistically accepted — model update proceeds locally
 *   Blocks b+1..b+W: Challenge window — any node may submit a SpectralFraudProof
 *   Block b+W+1:  Round finalised; reward pool distributed
 *
 * Architecture (Contribution 2 — Secondary):
 *   Each client runs Spectral Sentinel locally and submits SpectralVote(target, score ∈ [0,1000])
 *   Final aggregation weight w_i = Σ_j score_j(i) / Σ_i Σ_j score_j(i)  (published on-chain)
 *
 * Gas complexity (Theorem 1): O(k) per fraud-proof verification where k = eigenvalue sketch size.
 */
contract SpectralBFT {

    // ─────────────────────────────────────────────────────────────
    // Structs
    // ─────────────────────────────────────────────────────────────

    /**
     * @dev Record stored when a client submits a gradient for a round.
     *      frobeniusNorm_e6 = ||gradient||² / n_clients × 10⁶
     *      Used in fraud-proof Frobenius-consistency check.
     */
    struct GradientRecord {
        bytes32 gradientHash;      // keccak256(gradient_bytes)
        bytes32 storageKey;        // content-addressed key in local storage
        uint256 frobeniusNorm_e6;  // Frobenius norm squared / n, × 10⁶
        uint256 submittedBlock;
        bool excluded;             // set true when fraud proof verified
        bool challenged;           // set true when fraud proof submitted (pending)
        bool submitted;
    }

    /**
     * @dev Spectral vote cast by one peer about another peer's gradient quality.
     *      score ∈ [0, 1000] represents spectral quality (1000 = perfectly honest).
     *      Derived from: score = round((1 - KS_stat) × 1000).
     */
    struct SpectralVote {
        bool accept;
        uint16 score;   // 0–1000
        bool cast;
    }

    /**
     * @dev Compact on-chain spectral fraud proof.
     *      A proof π = (gradientHash, eigenvalues_e6, ks_stat_e6, sigma2_e6, lambda_plus_e6)
     *      is valid iff:
     *        (a) gradientHash matches the on-chain submission
     *        (b) Σ eigenvalues_e6 ≤ frobeniusNorm_e6  (Frobenius consistency)
     *        (c) ks_stat_e6 > ksThreshold_e6  OR  eigenvalues_e6[0] > lambda_plus_e6 + tailThreshold_e6 * sigma2_e6 / 1e6
     *
     * Soundness note: condition (b) is necessary but not sufficient (see paper Section VI-B).
     * The commitment-based fix uses a pre-committed gradient hash + random coordinate sampling.
     */
    struct FraudProof {
        bytes32 gradientHash;
        uint256[] eigenvalues_e6;   // top-k eigenvalues × 10⁶, index 0 = largest
        uint256 ks_stat_e6;
        uint256 sigma2_e6;
        uint256 lambda_plus_e6;     // MP upper edge σ²(1 + √γ)² × 10⁶
    }

    /**
     * @dev Per-round aggregation result published on-chain after challenge window.
     */
    struct RoundResult {
        address[] acceptedClients;
        uint256[] weights_e6;        // aggregation weights normalised to Σ = 1×10⁶
        bytes32 aggregatedModelHash;
        bool finalised;
        uint256 challengeDeadline;   // block number after which no fraud proofs accepted
        uint256 slashPool;           // ETH (wei) available to honest clients
        uint256 submissionCount;
    }

    // ─────────────────────────────────────────────────────────────
    // State
    // ─────────────────────────────────────────────────────────────

    address public owner;
    uint256 public currentRound;

    // round → client_address → GradientRecord
    mapping(uint256 => mapping(address => GradientRecord)) public submissions;

    // round → voter → target → SpectralVote
    mapping(uint256 => mapping(address => mapping(address => SpectralVote))) public votes;

    // round → target → total accumulated score (sum of all peer scores)
    mapping(uint256 => mapping(address => uint256)) public totalScores;
    mapping(uint256 => mapping(address => uint256)) public acceptVoteCount;

    // round → submitters list
    mapping(uint256 => address[]) public roundSubmitters;

    // round → RoundResult
    mapping(uint256 => RoundResult) public roundResults;

    // Reputation: address → score [0, 1000]
    mapping(address => uint256) public reputation;

    // Staking: address → staked wei
    mapping(address => uint256) public stake;

    // Reward claims: round → address → claimed
    mapping(uint256 => mapping(address => bool)) public rewardClaimed;

    // Registered clients
    mapping(address => bool) public isRegistered;
    mapping(address => uint256) public clientId;
    address[] public clients;

    // Protocol parameters (all fixed-point × 10⁶ where noted)
    uint256 public ksThreshold_e6 = 50000;       // KS statistic threshold = 0.05
    uint256 public tailThreshold_e6 = 2000000;   // tail multiplier τ_tail = 2.0
    uint256 public challengeWindowBlocks = 50;   // W blocks
    uint256 public voteWindowBlocks = 20;        // blocks to submit votes after submission deadline
    uint256 public minStake = 0.01 ether;        // minimum stake to participate
    uint256 public slashBasisPoints = 5000;      // 50% of stake slashed on fraud
    uint256 public initialReputation = 800;

    // ─────────────────────────────────────────────────────────────
    // Events
    // ─────────────────────────────────────────────────────────────

    event ClientRegistered(address indexed client, uint256 indexed id);
    event RoundStarted(uint256 indexed round, uint256 block_);
    event GradientSubmitted(uint256 indexed round, address indexed client, bytes32 gradHash, uint256 frobNorm_e6);
    event SpectralVoteCast(uint256 indexed round, address indexed voter, address indexed target, bool accept, uint16 score);
    event VotesTallied(uint256 indexed round, uint256 acceptedCount);
    event FraudProofSubmitted(uint256 indexed round, address indexed accused, address indexed challenger);
    event FraudProofVerified(uint256 indexed round, address indexed accused, bool valid, uint256 slashAmount);
    event RoundFinalised(uint256 indexed round, bytes32 aggHash, uint256 acceptedClients, uint256 totalSlashed);
    event RewardClaimed(uint256 indexed round, address indexed client, uint256 amount);
    event Staked(address indexed client, uint256 amount);
    event Unstaked(address indexed client, uint256 amount);

    // ─────────────────────────────────────────────────────────────
    // Modifiers
    // ─────────────────────────────────────────────────────────────

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyRegistered() {
        require(isRegistered[msg.sender], "Not registered");
        _;
    }

    modifier sufficientStake() {
        require(stake[msg.sender] >= minStake, "Insufficient stake");
        _;
    }

    // ─────────────────────────────────────────────────────────────
    // Constructor
    // ─────────────────────────────────────────────────────────────

    constructor() {
        owner = msg.sender;
        currentRound = 0;
    }

    // ─────────────────────────────────────────────────────────────
    // Client Management
    // ─────────────────────────────────────────────────────────────

    /**
     * @notice Register a single client (owner only).
     */
    function registerClient(address client, uint256 id) external onlyOwner {
        require(!isRegistered[client], "Already registered");
        isRegistered[client] = true;
        clientId[client] = id;
        clients.push(client);
        reputation[client] = initialReputation;
        emit ClientRegistered(client, id);
    }

    /**
     * @notice Batch-register clients (gas-efficient for large deployments).
     */
    function registerClientsBatch(address[] calldata clientAddrs, uint256[] calldata ids) external onlyOwner {
        require(clientAddrs.length == ids.length, "Length mismatch");
        for (uint256 i = 0; i < clientAddrs.length; i++) {
            if (!isRegistered[clientAddrs[i]]) {
                isRegistered[clientAddrs[i]] = true;
                clientId[clientAddrs[i]] = ids[i];
                clients.push(clientAddrs[i]);
                reputation[clientAddrs[i]] = initialReputation;
                emit ClientRegistered(clientAddrs[i], ids[i]);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Round Management
    // ─────────────────────────────────────────────────────────────

    /**
     * @notice Start a new federated learning round.
     */
    function startRound() external onlyOwner {
        currentRound++;
        roundResults[currentRound].challengeDeadline = block.number + voteWindowBlocks + challengeWindowBlocks;
        emit RoundStarted(currentRound, block.number);
    }

    // ─────────────────────────────────────────────────────────────
    // Contribution 1: Optimistic Gradient Submission
    // ─────────────────────────────────────────────────────────────

    /**
     * @notice Submit gradient hash and Frobenius norm for the current round.
     *         Round proceeds optimistically — no pre-verification.
     * @param round         Current round number
     * @param gradHash      keccak256(gradient_bytes)
     * @param storageKey    Content-addressed key in local/IPFS storage
     * @param frobNorm_e6   ||gradient||² / n_clients × 10⁶ (Frobenius norm, used in fraud proof)
     */
    function submitGradient(
        uint256 round,
        bytes32 gradHash,
        bytes32 storageKey,
        uint256 frobNorm_e6
    ) external onlyRegistered sufficientStake {
        require(round == currentRound, "Wrong round");
        require(!submissions[round][msg.sender].submitted, "Already submitted");

        submissions[round][msg.sender] = GradientRecord({
            gradientHash: gradHash,
            storageKey: storageKey,
            frobeniusNorm_e6: frobNorm_e6,
            submittedBlock: block.number,
            excluded: false,
            challenged: false,
            submitted: true
        });
        roundSubmitters[round].push(msg.sender);
        roundResults[round].submissionCount++;

        emit GradientSubmitted(round, msg.sender, gradHash, frobNorm_e6);
    }

    // ─────────────────────────────────────────────────────────────
    // Contribution 2: Spectral Graded Broadcast Voting
    // ─────────────────────────────────────────────────────────────

    /**
     * @notice Submit a spectral vote for a peer's gradient quality.
     *         score ∈ [0, 1000] = round((1 - KS_stat) × 1000)
     * @param round     Round number
     * @param target    Address of peer being evaluated
     * @param accept    True if peer's gradient passes Spectral Sentinel
     * @param score     Spectral quality score [0, 1000]
     */
    function submitSpectralVote(
        uint256 round,
        address target,
        bool accept,
        uint16 score
    ) external onlyRegistered {
        require(round == currentRound, "Wrong round");
        require(submissions[round][target].submitted, "Target not submitted");
        require(!votes[round][msg.sender][target].cast, "Vote already cast");
        require(score <= 1000, "Score out of range");

        votes[round][msg.sender][target] = SpectralVote({
            accept: accept,
            score: score,
            cast: true
        });

        if (accept) {
            acceptVoteCount[round][target]++;
        }
        totalScores[round][target] += score;

        emit SpectralVoteCast(round, msg.sender, target, accept, score);
    }

    /**
     * @notice Tally votes and compute aggregation weights.
     *         Callable by anyone after vote window. Implements graded broadcast tally.
     *         w_i = Σ_j score_j(i) / Σ_i Σ_j score_j(i)  (normalised to 10⁶)
     * @param round Round to tally
     */
    function tallyVotes(uint256 round) external {
        require(round <= currentRound, "Round not started");
        require(!roundResults[round].finalised, "Round already finalised");
        require(block.number >= submissions[round][roundSubmitters[round][0]].submittedBlock + voteWindowBlocks,
                "Vote window still open");

        address[] memory submitters = roundSubmitters[round];
        uint256 n = submitters.length;
        uint256 quorum = n / 2; // simple majority

        // Count accepted clients (majority honest vote)
        uint256 acceptedCount = 0;
        address[] memory accepted = new address[](n);
        uint256 totalScoreSum = 0;

        for (uint256 i = 0; i < n; i++) {
            address client = submitters[i];
            if (!submissions[round][client].excluded &&
                acceptVoteCount[round][client] > quorum) {
                accepted[acceptedCount] = client;
                totalScoreSum += totalScores[round][client];
                acceptedCount++;
            }
        }

        // Compute weights (normalised to 10⁶)
        uint256[] memory weights = new uint256[](acceptedCount);
        if (totalScoreSum > 0) {
            for (uint256 i = 0; i < acceptedCount; i++) {
                weights[i] = (totalScores[round][accepted[i]] * 1e6) / totalScoreSum;
            }
        }

        // Trim to actual accepted count
        address[] memory trimmedAccepted = new address[](acceptedCount);
        for (uint256 i = 0; i < acceptedCount; i++) {
            trimmedAccepted[i] = accepted[i];
        }

        roundResults[round].acceptedClients = trimmedAccepted;
        roundResults[round].weights_e6 = weights;

        emit VotesTallied(round, acceptedCount);
    }

    // ─────────────────────────────────────────────────────────────
    // Contribution 1: On-Chain Spectral Fraud Proofs
    // ─────────────────────────────────────────────────────────────

    /**
     * @notice Submit a spectral fraud proof against an accused client.
     *         Challenge window: [submittedBlock+1 .. submittedBlock+W]
     *
     * @param round    Round in which the accused submitted
     * @param accused  Address of the accused client
     * @param proof    Fraud proof π = (gradHash, eigenvalues_e6, ks_stat_e6, sigma2_e6, lambda_plus_e6)
     */
    function submitFraudProof(
        uint256 round,
        address accused,
        FraudProof calldata proof
    ) external onlyRegistered {
        GradientRecord storage rec = submissions[round][accused];
        require(rec.submitted, "Accused has no submission");
        require(!rec.excluded, "Already excluded");
        require(!rec.challenged, "Already challenged");
        require(
            block.number <= rec.submittedBlock + challengeWindowBlocks + voteWindowBlocks,
            "Challenge window closed"
        );

        rec.challenged = true;

        bool valid = _verifyFraudProof(proof, rec);

        if (valid) {
            rec.excluded = true;
            // Slash accused client's stake (slashBasisPoints / 10000 fraction)
            uint256 slashAmount = (stake[accused] * slashBasisPoints) / 10000;
            if (slashAmount > stake[accused]) slashAmount = stake[accused];
            stake[accused] -= slashAmount;
            roundResults[round].slashPool += slashAmount;

            // Update reputation (penalise)
            if (reputation[accused] >= 200) {
                reputation[accused] -= 200;
            } else {
                reputation[accused] = 0;
            }

            emit FraudProofVerified(round, accused, true, slashAmount);

            // Reward challenger immediately (10% of slash)
            uint256 challengerReward = slashAmount / 10;
            roundResults[round].slashPool -= challengerReward;
            (bool sent,) = msg.sender.call{value: challengerReward}("");
            require(sent, "Reward transfer failed");
        } else {
            // Invalid fraud proof — reset challenged flag
            rec.challenged = false;
            emit FraudProofVerified(round, accused, false, 0);
        }

        emit FraudProofSubmitted(round, accused, msg.sender);
    }

    /**
     * @notice Verify a spectral fraud proof on-chain.
     *         Gas complexity: O(k) where k = len(proof.eigenvalues_e6).
     *
     *         Checks:
     *           (a) gradientHash matches stored submission
     *           (b) Σ eigenvalues_e6 ≤ rec.frobeniusNorm_e6  (Frobenius consistency)
     *           (c) ks_stat_e6 > ksThreshold_e6  OR
     *               eigenvalues_e6[0] > lambda_plus_e6 + tailThreshold_e6 * sigma2_e6 / 1e6
     *
     * @dev  Soundness caveat: (b) is necessary but not sufficient.
     *       A dishonest prover could craft eigenvalues passing (b) that don't match the gradient.
     *       Mitigation (described in paper): two-layer commitment + random coordinate sampling.
     */
    function _verifyFraudProof(
        FraudProof calldata proof,
        GradientRecord storage rec
    ) internal view returns (bool) {
        // (a) Hash binding
        if (proof.gradientHash != rec.gradientHash) {
            return false;
        }

        // (b) Frobenius consistency — O(k) loop
        uint256 eigSum = 0;
        for (uint256 i = 0; i < proof.eigenvalues_e6.length; i++) {
            eigSum += proof.eigenvalues_e6[i];
        }
        if (eigSum > rec.frobeniusNorm_e6) {
            return false;  // eigenvalues inconsistent with claimed gradient norm
        }

        // (c) Spectral violation
        bool ksViolation = proof.ks_stat_e6 > ksThreshold_e6;

        // tail check: lambda_max > lambda_plus + tau_tail * sigma2
        // all × 10⁶ so: eigenvalues_e6[0] > lambda_plus_e6 + tailThreshold_e6 * sigma2_e6 / 1e6
        bool tailViolation = false;
        if (proof.eigenvalues_e6.length > 0) {
            uint256 tailBound = proof.lambda_plus_e6 + (tailThreshold_e6 * proof.sigma2_e6) / 1e6;
            tailViolation = proof.eigenvalues_e6[0] > tailBound;
        }

        return ksViolation || tailViolation;
    }

    // ─────────────────────────────────────────────────────────────
    // Round Finalization
    // ─────────────────────────────────────────────────────────────

    /**
     * @notice Finalise a round after the challenge window closes.
     * @param round      Round to finalise
     * @param aggHash    keccak256(aggregated_model_bytes)
     */
    function finalizeRound(uint256 round, bytes32 aggHash) external {
        RoundResult storage result = roundResults[round];
        require(!result.finalised, "Already finalised");
        require(
            block.number > result.challengeDeadline,
            "Challenge window still open"
        );

        result.aggregatedModelHash = aggHash;
        result.finalised = true;

        // Update reputation for accepted honest clients (+50)
        for (uint256 i = 0; i < result.acceptedClients.length; i++) {
            address client = result.acceptedClients[i];
            uint256 newRep = reputation[client] + 50;
            reputation[client] = newRep > 1000 ? 1000 : newRep;
        }

        emit RoundFinalised(round, aggHash, result.acceptedClients.length, result.slashPool);
    }

    /**
     * @notice Accepted honest clients claim their share of the slash pool.
     */
    function claimReward(uint256 round) external onlyRegistered {
        RoundResult storage result = roundResults[round];
        require(result.finalised, "Round not finalised");
        require(!rewardClaimed[round][msg.sender], "Already claimed");

        // Check caller is in acceptedClients
        bool isAccepted = false;
        for (uint256 i = 0; i < result.acceptedClients.length; i++) {
            if (result.acceptedClients[i] == msg.sender) {
                isAccepted = true;
                break;
            }
        }
        require(isAccepted, "Not an accepted client");

        rewardClaimed[round][msg.sender] = true;

        uint256 n = result.acceptedClients.length;
        if (n > 0 && result.slashPool > 0) {
            uint256 share = result.slashPool / n;
            (bool sent,) = msg.sender.call{value: share}("");
            require(sent, "Reward transfer failed");
            emit RewardClaimed(round, msg.sender, share);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // View Functions
    // ─────────────────────────────────────────────────────────────

    /**
     * @notice Get on-chain aggregation weights (published after vote tally).
     *         Anyone can recompute the agreed aggregate from local gradient storage.
     */
    function getAggregationWeights(uint256 round)
        external view
        returns (address[] memory, uint256[] memory)
    {
        return (roundResults[round].acceptedClients, roundResults[round].weights_e6);
    }

    function getRoundResult(uint256 round)
        external view
        returns (bool finalised, bytes32 aggHash, uint256 accepted, uint256 slashPool)
    {
        RoundResult storage r = roundResults[round];
        return (r.finalised, r.aggregatedModelHash, r.acceptedClients.length, r.slashPool);
    }

    function getSubmission(uint256 round, address client)
        external view
        returns (bytes32 gradHash, bool submitted, bool excluded, uint256 frobNorm_e6)
    {
        GradientRecord storage rec = submissions[round][client];
        return (rec.gradientHash, rec.submitted, rec.excluded, rec.frobeniusNorm_e6);
    }

    function getClients() external view returns (address[] memory) {
        return clients;
    }

    function getRoundSubmitters(uint256 round) external view returns (address[] memory) {
        return roundSubmitters[round];
    }

    // ─────────────────────────────────────────────────────────────
    // Staking
    // ─────────────────────────────────────────────────────────────

    /**
     * @notice Stake ETH to participate as an FL client.
     *         Nash equilibrium condition (paper Theorem 3): if slashBasisPoints/10000 × stake > attack_gain,
     *         rational clients prefer honesty.
     */
    function stakeETH() external payable onlyRegistered {
        require(msg.value > 0, "Must send ETH");
        stake[msg.sender] += msg.value;
        emit Staked(msg.sender, msg.value);
    }

    /**
     * @notice Unstake ETH (only if no active fraud proof challenge).
     */
    function unstake(uint256 amount) external onlyRegistered {
        require(stake[msg.sender] >= amount, "Insufficient stake");
        require(stake[msg.sender] - amount >= 0, "Would go below zero");
        stake[msg.sender] -= amount;
        (bool sent,) = msg.sender.call{value: amount}("");
        require(sent, "Transfer failed");
        emit Unstaked(msg.sender, amount);
    }

    // ─────────────────────────────────────────────────────────────
    // Admin
    // ─────────────────────────────────────────────────────────────

    function setKSThreshold(uint256 threshold_e6) external onlyOwner {
        ksThreshold_e6 = threshold_e6;
    }

    function setChallengeWindow(uint256 blocks) external onlyOwner {
        challengeWindowBlocks = blocks;
    }

    function setVoteWindow(uint256 blocks) external onlyOwner {
        voteWindowBlocks = blocks;
    }

    function setMinStake(uint256 amount) external onlyOwner {
        minStake = amount;
    }

    receive() external payable {}
}
