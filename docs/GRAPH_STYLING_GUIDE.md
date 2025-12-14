# Graph Styling Reference Guide

## Quick Reference for report.tex Graphs

### Standard Axis Configuration

```latex
\begin{axis}[
    xlabel={Descriptive Label},
    ylabel={Metric (Unit)},
    xmin=..., xmax=...,
    ymin=..., ymax=...,
    grid=major,
    grid style={dashed, gray!20},
    width=0.48\textwidth,
    height=0.35\textwidth,
    legend pos=south east,  % or north west/east as appropriate
    legend style={font=\scriptsize, draw=black!50, fill=white, fill opacity=0.9, text opacity=1},
    tick label style={font=\footnotesize},
    xlabel style={font=\small},
    ylabel style={font=\small},
    minor tick num=1
]
```

---

## Color Palette

### Primary Colors (Main Results)
```latex
blue!70!black     % Our method - trustworthy, professional
red!70!black      % Best baseline or critical points
green!60!black    % Efficient alternatives
purple!70!black   % Additional methods
orange!80!black   % Warning/problematic regions
```

### Secondary Colors (Reference/Context)
```latex
black!40          % Reference lines, subtle annotations
black!50          % Secondary annotations
gray              % Grid lines (gray!20 for background)
```

### Background Shading
```latex
blue!10, opacity=0.3      % Positive/safe regions
red!10, opacity=0.3       % Critical/transition regions  
orange!10, opacity=0.3    % Warning/dangerous regions
green!5, opacity=0.5      % Recommended/practical ranges
```

---

## Line Styling

### Our Method (Emphasis)
```latex
\addplot[color=blue!70!black, line width=2pt] coordinates {...};
% or for extra emphasis:
\addplot[color=blue!70!black, line width=2.5pt, mark=*, mark size=3pt, mark options={fill=blue!70!black}] coordinates {...};
```

### Best Baseline
```latex
\addplot[color=red!70!black, line width=1.5pt, dashed] coordinates {...};
```

### Other Baselines
```latex
\addplot[color=green!60!black, line width=1.2pt, dashdotted] coordinates {...};
\addplot[color=purple!70!black, line width=1.2pt, densely dotted] coordinates {...};
```

### Reference Lines
```latex
\addplot[black!40, line width=1pt, densely dotted] coordinates {...};
```

---

## Marker Styles

### Filled Markers (Recommended)
```latex
mark=*              % Filled circle
mark=square*        % Filled square
mark=triangle*      % Filled triangle
mark=diamond*       % Filled diamond
mark options={fill=blue!70!black}  % Match line color
```

### Marker Sizes
```latex
mark size=3pt       % Primary data (our method)
mark size=2.5pt     % Important comparisons
mark size=2pt       % Standard data points
```

---

## Annotation Styles

### Critical Points/Boundaries
```latex
\draw[line width=2.5pt, red!60!black, densely dashed] (axis cs:X,Ymin) -- (axis cs:X,Ymax);
\node[red!60!black, font=\footnotesize\bfseries, fill=white, inner sep=2pt, draw=red!60!black, rounded corners] at (axis cs:X,Y) {Label};
```

### Region Labels
```latex
\node[blue!70!black, font=\scriptsize\bfseries, fill=white, inner sep=2pt, rounded corners, opacity=0.9] at (axis cs:X,Y) {Region Name};
```

### Data Point Labels
```latex
\node[font=\tiny, anchor=south] at (axis cs:X,Y) {Model Name};
```

### Reference Line Labels
```latex
\node[black!50, font=\tiny, fill=white, inner sep=1pt] at (axis cs:X,Y) {Reference};
```

### Rotated Labels (Vertical Guidelines)
```latex
\node[green!60!black, font=\scriptsize\bfseries, fill=white, inner sep=2pt, anchor=south, rotate=90] at (axis cs:X,Y) {Label};
```

---

## Background Elements

### Shaded Regions
```latex
\fill[blue!10, opacity=0.3] (axis cs:Xmin,Ymin) rectangle (axis cs:Xmax,Ymax);
```

### Reference Lines (Horizontal)
```latex
\draw[black!30, densely dotted, line width=1.5pt] (axis cs:Xmin,Y) -- (axis cs:Xmax,Y);
```

### Reference Lines (Vertical)
```latex
\draw[thick, green!60!black, densely dashed] (axis cs:X,Ymin) -- (axis cs:X,Ymax);
```

---

## Legend Configuration

### Standard Legend
```latex
\legend{Clean (No Attack), \textbf{Our Method (Ours)}, Baseline 1, Baseline 2}
```

### Legend with Descriptions
```latex
\legend{Full Cov. ($O(d^2)$), Sketch $k$=256, \textbf{Sketch $k$=512 (Ours)}, Sketch $k$=1024}
```

### Legend Position
```latex
legend pos=south east    % For convergence/increasing trends
legend pos=north west    % For memory/decreasing trends  
legend pos=north east    % For phase transitions
```

---

## Grid Styling

### Standard Grid
```latex
grid=major,
grid style={dashed, gray!20},
minor tick num=1
```

### Enhanced Grid (Log Scales)
```latex
grid=both,
grid style={line width=0.1pt, draw=gray!20},
major grid style={line width=0.3pt, draw=gray!40},
minor tick num=9
```

---

## Axis Formatting

### Standard Linear Axis
```latex
xmin=0, xmax=200,
ymin=45, ymax=88,
xtick={0,50,100,150,200},
ytick={50,60,70,80}
```

### Log Axis (Readable Labels)
```latex
xmode=log,
ymode=log,
log basis x=10,
log basis y=10,
xtick={1e5,1e6,1e7,1e8,1e9},
xticklabels={100K,1M,10M,100M,1B},
ytick={1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3},
yticklabels={0.001,0.01,0.1,1,10,100,1000}
```

---

## Caption Style

### Format
```latex
\caption{Main description: Specific quantitative result (metric values), comparing our method vs baselines. Additional context about significance.}
```

### Examples

**Good:**
```latex
\caption{Convergence under Byzantine attacks (40\% adversarial clients): Spectral Sentinel achieves 79.1\% accuracy vs 63.4\% for best baseline (FLTrust), approaching clean performance of 85.0\%.}
```

**Good:**
```latex
\caption{Memory scaling comparison: Sketching reduces memory from $O(d^2)$ to $O(k^2)$, enabling 345M parameter models with 890MB vs 28GB (31$\times$ reduction).}
```

**Avoid:**
```latex
\caption{Convergence curves showing different methods.}  % Too vague!
```

---

## Common Patterns

### Pattern 1: Convergence Curves
1. Show clean/optimal baseline as reference (dotted, light color)
2. Emphasize our method (thick blue line, 2pt)
3. Show best baseline (dashed red, 1.5pt)
4. Show other methods (various styles, 1.2pt)
5. Legend lists clean first, then our method in bold

### Pattern 2: Phase Transitions
1. Shade background regions
2. Use distinct markers for each regime
3. Draw thick dashed line at critical point
4. Add labeled box at transition
5. Include region labels

### Pattern 3: Scaling Analysis
1. Use log scales for wide ranges
2. Add reference lines (GPU memory, saturation, etc.)
3. Annotate key model sizes
4. Emphasize recommended configuration
5. Show full curve including impractical regions

### Pattern 4: Parameter Sensitivity
1. Shade practical operating range
2. Add saturation/limit lines
3. Mark recommended values with vertical lines
4. Use rotated labels for recommendations
5. Show multiple configurations for comparison

---

## Typography Hierarchy

```latex
\footnotesize          % Tick labels (x/y axis numbers)
\small                 % Axis labels (xlabel, ylabel)
\scriptsize            % Legend text
\scriptsize\bfseries   % Important annotations
\tiny                  % Secondary annotations, notes
```

---

## Quick Checklist

Before finalizing any graph:

✅ Is our method emphasized? (Thickest line, bold in legend)  
✅ Are colors professional? (color!70!black format)  
✅ Is the grid subtle? (dashed, gray!20)  
✅ Are markers filled? (Visible in grayscale)  
✅ Is the legend clear? (Semi-transparent background, ordered)  
✅ Are annotations useful? (Bordered boxes, good placement)  
✅ Is the caption quantitative? (Specific numbers, comparisons)  
✅ Are references shown? (GPU limits, baselines, etc.)  
✅ Is text readable? (Appropriate font sizes)  
✅ Does it tell a story? (Guides reader to conclusion)  

---

## Don't Do This!

❌ Basic colors: `blue`, `red`, `green`  
✅ Professional: `blue!70!black`, `red!70!black`, `green!60!black`

❌ All same line width  
✅ Hierarchical: 2.5pt → 2pt → 1.5pt → 1.2pt

❌ Solid grid  
✅ Dashed, subtle: `grid style={dashed, gray!20}`

❌ Empty markers  
✅ Filled: `mark=*, mark options={fill=...}`

❌ Plain text annotations  
✅ Bordered boxes: `fill=white, draw=..., rounded corners`

❌ Generic caption: "Results for different methods"  
✅ Quantitative: "79.1% vs 63.4% for best baseline"

---

## File Organization

```
report.tex
├── Packages (lines 1-23)
├── Content
└── Figures
    ├── Fig 1: Convergence (lines 415-461)
    ├── Fig 2: Phase Transition (lines 467-537)
    ├── Fig 3: Memory Scaling (lines 545-632)
    └── Fig 4: Sketch Size (lines 766-822)
```

---

## Compilation

```bash
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex  # Run twice for references
```

If you get errors, check:
1. All `{` have matching `}`
2. All `\begin{axis}` have `\end{axis}`
3. All coordinates have matching parentheses
4. Color definitions are valid (e.g., `blue!70!black`)

---

## Summary

**Key Principle**: Every visual element should serve a purpose and guide the reader toward understanding the research contribution.

**Our Philosophy**:
- **Emphasize** what matters (our method)
- **Provide** context (baselines, references)
- **Guide** the reader (annotations, shading)
- **Quantify** results (captions)
- **Look** professional (colors, styling)

Follow these guidelines for consistent, publication-quality figures!
