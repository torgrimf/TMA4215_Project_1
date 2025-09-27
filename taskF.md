Note that the approximation of the 2-norm of the error
$$
\left\| f-p_{n}  \right\| _2 = \sqrt{\frac{b-a}{N}} \left( \sum_{k=0}^{N} \left( f(\eta_k)-p_n(\eta_k)^2 \right)  \right) ^{1 /2}
$$
is exactly $\sqrt{\mathcal{C}\left( \mathbf{x} \right)}  $. So minimizing $\mathcal{C}(\mathbf{x})$ minimizes the 2-norm of the error squared when $N \to \infty$, and consequently minimizes the 2-norm of the error (not squared because sqrt is injective). So by increasing $N$ we will get closer to an approximation of the actual norm of the error. I.e. in the limit 
$$
\lim_{N \to \infty} \mathcal{C}(\mathbf{x}) = \left\| f-p_{n}  \right\|_2. 
$$

Assume we find the optimal nodes $\mathbf{x}$, then we have found a $p_n$ such that $\mathcal{C}(\mathbf{x})$ is minimized, i.e.
$$
\min _{p_n \in \Pi_n} \mathcal{C}(\mathbf{x}) = \min _{p_{n} \in \Pi_n} \left\| f -p_n \right\|_2^2.
$$
But remember that the best approximation polynomial $\widetilde{p}_n$ in the 2-norm is the unique $p_n \in \Pi_n$ that satisfies
$$
\left\| f- \widetilde{p}_n \right\|_2= \min _{q \in \Pi_n} \left\| f- q \right\| _2.
$$

So if $p_n$ is then exactly the $q$ that minimizes the 2-norm of the error, then the interpolation polynomial $p_n$ constructed on the optimal nodes must be the best approximation polynomial $\tilde{p}_n$.

*For the second question.*

Seeking coefficients $c_{0},...,c_n$ to minimize the 2-norm of the error such that 
$$
p_n(x) = c_{0}+c_{1}x+...+ c_nx^{n}
$$
leads to a linear system $Mc=b$ to be solved. Here $M$ is $(n+1)\times(n+1)$ and $M_{jk}=\left<	x^{k},\,x^{j}  \right>$, $b_j = \left<	f,\,x^{j}  \right>$. This matrix is ill conditioned and becomes unpractical to solve as $n$ increases. A better method would be to seek a diagonal matrix, namely by constructing an orthogonal basis. In this case that would be the Legendre polynomials.