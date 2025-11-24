// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FederatedLearning
 * @notice Smart contract for blockchain-enabled federated learning
 * @dev Stores model update hashes and coordinates federated learning rounds
 */
contract FederatedLearning {
    
    // ==================== Structs ====================
    
    struct ModelUpdate {
        address clientAddress;
        uint256 clientId;
        bytes32 modelHash;
        uint256 timestamp;
        uint256 roundNumber;
        bool verified;
    }
    
    struct Round {
        uint256 roundNumber;
        uint256 startTime;
        uint256 endTime;
        bytes32 aggregatedModelHash;
        uint256 numSubmissions;
        bool finalized;
    }
    
    // ==================== State Variables ====================
    
    address public owner;
    uint256 public currentRound;
    uint256 public totalClients;
    
    // Mappings
    mapping(uint256 => address) public clientIdToAddress;
    mapping(address => uint256) public addressToClientId;
    mapping(address => bool) public isRegisteredClient;
    
    // Round number => client ID => ModelUpdate
    mapping(uint256 => mapping(uint256 => ModelUpdate)) public roundUpdates;
    
    // Round number => Round info
    mapping(uint256 => Round) public rounds;
    
    // Track submissions per round per client
    mapping(uint256 => mapping(uint256 => bool)) public hasSubmitted;
    
    // ==================== Events ====================
    
    event ClientRegistered(address indexed clientAddress, uint256 clientId);
    event RoundStarted(uint256 roundNumber, uint256 timestamp);
    event ModelUpdateSubmitted(
        uint256 indexed roundNumber,
        uint256 indexed clientId,
        address indexed clientAddress,
        bytes32 modelHash,
        uint256 timestamp
    );
    event RoundFinalized(
        uint256 indexed roundNumber,
        bytes32 aggregatedModelHash,
        uint256 numSubmissions,
        uint256 timestamp
    );
    
    // ==================== Modifiers ====================
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyRegisteredClient() {
        require(isRegisteredClient[msg.sender], "Client not registered");
        _;
    }
    
    modifier roundActive(uint256 _roundNumber) {
        require(rounds[_roundNumber].startTime > 0, "Round not started");
        require(!rounds[_roundNumber].finalized, "Round already finalized");
        _;
    }
    
    // ==================== Constructor ====================
    
    constructor() {
        owner = msg.sender;
        currentRound = 0;
    }
    
    // ==================== Client Management ====================
    
    /**
     * @notice Register a new client for federated learning
     * @param _clientAddress Address of the client to register
     * @param _clientId Unique identifier for the client
     */
    function registerClient(address _clientAddress, uint256 _clientId) 
        external 
        onlyOwner 
    {
        require(!isRegisteredClient[_clientAddress], "Client already registered");
        require(clientIdToAddress[_clientId] == address(0), "Client ID already taken");
        
        clientIdToAddress[_clientId] = _clientAddress;
        addressToClientId[_clientAddress] = _clientId;
        isRegisteredClient[_clientAddress] = true;
        totalClients++;
        
        emit ClientRegistered(_clientAddress, _clientId);
    }
    
    /**
     * @notice Batch register multiple clients
     * @param _clientAddresses Array of client addresses
     * @param _clientIds Array of client IDs
     */
    function registerClientsBatch(
        address[] calldata _clientAddresses,
        uint256[] calldata _clientIds
    ) 
        external 
        onlyOwner 
    {
        require(_clientAddresses.length == _clientIds.length, "Array length mismatch");
        
        for (uint256 i = 0; i < _clientAddresses.length; i++) {
            if (!isRegisteredClient[_clientAddresses[i]] && 
                clientIdToAddress[_clientIds[i]] == address(0)) {
                clientIdToAddress[_clientIds[i]] = _clientAddresses[i];
                addressToClientId[_clientAddresses[i]] = _clientIds[i];
                isRegisteredClient[_clientAddresses[i]] = true;
                totalClients++;
                
                emit ClientRegistered(_clientAddresses[i], _clientIds[i]);
            }
        }
    }
    
    // ==================== Round Management ====================
    
    /**
     * @notice Start a new federated learning round
     */
    function startRound() external onlyOwner {
        if (currentRound > 0) {
            require(rounds[currentRound].finalized, "Previous round not finalized");
        }
        
        currentRound++;
        rounds[currentRound] = Round({
            roundNumber: currentRound,
            startTime: block.timestamp,
            endTime: 0,
            aggregatedModelHash: bytes32(0),
            numSubmissions: 0,
            finalized: false
        });
        
        emit RoundStarted(currentRound, block.timestamp);
    }
    
    /**
     * @notice Submit model update for current round
     * @param _modelHash Hash of the model update (e.g., SHA256 of serialized model)
     */
    function submitModelUpdate(bytes32 _modelHash) 
        external 
        onlyRegisteredClient 
        roundActive(currentRound)
    {
        uint256 clientId = addressToClientId[msg.sender];
        require(!hasSubmitted[currentRound][clientId], "Already submitted for this round");
        require(_modelHash != bytes32(0), "Invalid model hash");
        
        ModelUpdate memory update = ModelUpdate({
            clientAddress: msg.sender,
            clientId: clientId,
            modelHash: _modelHash,
            timestamp: block.timestamp,
            roundNumber: currentRound,
            verified: true
        });
        
        roundUpdates[currentRound][clientId] = update;
        hasSubmitted[currentRound][clientId] = true;
        rounds[currentRound].numSubmissions++;
        
        emit ModelUpdateSubmitted(
            currentRound,
            clientId,
            msg.sender,
            _modelHash,
            block.timestamp
        );
    }
    
    /**
     * @notice Finalize current round with aggregated model
     * @param _aggregatedModelHash Hash of the aggregated model
     */
    function finalizeRound(bytes32 _aggregatedModelHash) 
        external 
        onlyOwner 
        roundActive(currentRound)
    {
        require(_aggregatedModelHash != bytes32(0), "Invalid aggregated hash");
        
        rounds[currentRound].aggregatedModelHash = _aggregatedModelHash;
        rounds[currentRound].endTime = block.timestamp;
        rounds[currentRound].finalized = true;
        
        emit RoundFinalized(
            currentRound,
            _aggregatedModelHash,
            rounds[currentRound].numSubmissions,
            block.timestamp
        );
    }
    
    // ==================== View Functions ====================
    
    /**
     * @notice Get model update for a specific client in a round
     * @param _roundNumber Round number
     * @param _clientId Client ID
     * @return ModelUpdate struct
     */
    function getModelUpdate(uint256 _roundNumber, uint256 _clientId) 
        external 
        view 
        returns (ModelUpdate memory) 
    {
        return roundUpdates[_roundNumber][_clientId];
    }
    
    /**
     * @notice Get round information
     * @param _roundNumber Round number
     * @return Round struct
     */
    function getRoundInfo(uint256 _roundNumber) 
        external 
        view 
        returns (Round memory) 
    {
        return rounds[_roundNumber];
    }
    
    /**
     * @notice Check if client has submitted for a round
     * @param _roundNumber Round number
     * @param _clientId Client ID
     * @return Boolean indicating submission status
     */
    function hasClientSubmitted(uint256 _roundNumber, uint256 _clientId) 
        external 
        view 
        returns (bool) 
    {
        return hasSubmitted[_roundNumber][_clientId];
    }
    
    /**
     * @notice Get number of submissions for a round
     * @param _roundNumber Round number
     * @return Number of submissions
     */
    function getRoundSubmissions(uint256 _roundNumber) 
        external 
        view 
        returns (uint256) 
    {
        return rounds[_roundNumber].numSubmissions;
    }
    
    /**
     * @notice Get client ID from address
     * @param _clientAddress Client address
     * @return Client ID
     */
    function getClientId(address _clientAddress) 
        external 
        view 
        returns (uint256) 
    {
        require(isRegisteredClient[_clientAddress], "Client not registered");
        return addressToClientId[_clientAddress];
    }
}
