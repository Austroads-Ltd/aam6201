# Integer Programming Formulation

## Variables:

Let $N$ be the size of the dataset.
We have $N$ variables:
$$x_i \in \{0, 1\} \text{ for } i \in \{1, 2, ..., N\} $$

Treatment $i^{th}$ is selected if $x_i = 1$, not selected if $x_i = 0$

Cost of performing the selected treatment is then:
$$x_i * C_i$$
where $C_i$ is the cost of doing treatment for row $i$

## Budget constraint: 
$$\sum_{i} (x_i * C_i) \le \text{budget} $$

## Objective:
### Level of Service (LoS) gain/loss:
* Let $\text{delta\_pci\_treat} = \text{PCI\_after\_with\_treatment} - \text{PCI\_before}$
* Let $\text{delta\_pci\_no\_treat} = \text{PCI\_after\_without\_treatment} - \text{PCI\_before}$
* We have $\text{delta\_pci\_treat} \ge \text{delta\_pci\_no\_treat}$

### LoS term per $i$:
$$x_i * \text{delta\_pci\_treat}_i + (x_i * -\text{delta\_pci\_no\_treat}_i + \text{delta\_pci\_no\_treat}_i)$$
Evaluate above at:
* $x_i = 1$ gives $\text{delta\_pci\_treat}$
* $x_i = 0$ gives $\text{delta\_pci\_no\_treat}$

as required

### Rural/Urban Penalty:
* Let $u_i = 1$ if road section $i$ is a Metro road.
* Let $-100 <= p <= 100$
    * if $p = -100$, we expect there to be no rural 
    * if $p = 0$, we expect there to be no penalty
    * if $p = 100$, we expect there to be no metro 
* We have a penalty term for each metro/rural road we select to give a treatment too.
* $\text{penalty\_urban} = 0$ if $p \ge 0$ else $-|2 * p|$
* $\text{penalty\_rural} = 0$ if $p \le 0$ else $-|2 * p|$

**200 is larger than the maximum gain for choosing any treatment (gain by choosing treatment $-$ loss by not choosing treatment). Therefore, at $p = 100$ or $-100$, the penalty for choosing a metro/rural road outweights any possible gains.**

#### Penalty per $i$ if $x_i = 2$ 
$$\text{penalty}_i = \text{penalty\_urban} * u_i + \text{penalty\_rural} * (1 - u_i)$$

Penalty coefficient for each $i$:
$$ x_i * \text{penalty}_i$$ 
This is the same for all for other splits

### Objective full:

$$
\begin{align*}
\text{max} \\
&\sum_{i} x_i * \text{delta\_pci\_treat}_i + (x_i * -\text
{delta\_pci\_no\_treat}_i + \text{delta\_pci\_no\_treat}_i) \\
+ &\sum_{i} x_i * \text{penalty}_i \end{align*}
$$

is equivalent to

$$
\begin{align*}
\text{max}\\
&\sum_{i} x_i * (\text{delta\_pci\_treat}_i - \text{delta\_pci\_no\_treat}_i)  \\
+ &\sum_{i} x_i * \text{penalty}_i \\
+ &\sum_{i} \text{delta\_pci\_no\_treat}_i \\
\end{align*}
$$

or 

$$
\begin{align*}
\text{max}\\
&\sum_{i} x_i * (\text{delta\_pci\_treat}_i - \text{delta\_pci\_no\_treat}_i + \text{penalty}_i)\\
        + &\space C
\end{align*}
$$

The objective can be shortened to:
$$\text{max  } \sum_{i} x_i * (\text{delta\_pci\_treat}_i - \text{delta\_pci\_no\_treat}_i + \text{penalty}_i)$$

### (Optional) Split as hard constraint
Urban/Rural constraint: percentage of budget used on metro road must be larger than $l$ and smaller than $h$.

$$
\begin{align*}
    &\frac{\sum_{i} x_i * u_i * C_i}{\sum_{i} x_i * C_i} \gt l \\
\Longleftrightarrow &\sum_{i} x_i * u_i * C_i > \sum_{i} l * x_i * C_i \\
\Longleftrightarrow &\sum_{i} x_i * C_i * (u_i - l) > 0 
\end{align*}
$$
Switching $\gt l$ with $\lt h$ and combine the two equations, we have the hard constraint. We have the same equation for freight/non-freight split.
