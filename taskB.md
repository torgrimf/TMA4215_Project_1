We have $f(x) = \cos(2\pi x)$. With  
$$
\lim_{n \to \infty} \| f(x) - p_n(x) \|_\infty = 0
$$  

and similarly for max norm on both equidistant nodes and Chebyshev nodes on $[0,1]$.  



Thm 6.10 in Süli & Mayers gives  

$$
\| f(x) - p_n(x) \|_\infty \le \frac{M_{n+1}}{(n+1)!} \max_{x \in [a,b]} | \pi_{n+1}(x) |
$$  

if $f$ is continuous and real valued on the closed interval $[a,b]$. $f(x) = \cos(2\pi x) \in C^\infty[0,1]$, so the theorem holds. Then  

$$
\lim_{n \to \infty} \max_{x \in [0,1]} | f(x) - p_n(x) | 
\le \lim_{n \to \infty} \frac{M_{n+1}}{(n+1)!} \max_{x \in [0,1]} | \pi_{n+1}(x) |
$$  

So we have convergence if right side is equal $0$.  With $M_{n+1} = \max_{\zeta \in [a,b]} \left| f ^{n+1}(\zeta) \right|  $ it is
$$ 
\max_{\zeta \in [0,1]} \left| \frac{d^{n+1}}{d \zeta^{n+1}} \cos(2\pi \zeta) \right| \le (2\pi)^{n+1}
$$.  

since $\cos(x)$ is bounded by $[-1,1]$. Then  

$$
\frac{M_{n+1}}{(n+1)!} \max_{x \in [0,1]} | \pi_{n+1}(x) | 
\le \frac{(2\pi)^{n+1}}{(n+1)!} \max_{x \in [0,1]} | \pi_{n+1}(x)  |
$$  


With equidistant nodes use that  
$$
\left| \pi_{n+1}(x) \right| \le \left( \frac{b -a}{n} \right)^{n+1}  \cdot \frac{n!}{4}
$$

Then  

$$
\begin{align*}
    \frac{M_{n+1}}{(n+1)!} \max_{x \in [0,1]} \left| \pi_{n+1}(x) \right| 
&\le \frac{(2\pi)^{n+1}}{(n+1)!} \cdot  \left( \frac{1-0}{n} \right)^{n+1}  \cdot \frac{n!}{4}\\
&= \frac{(2\pi)^{n+1}}{4n^{n+1}(n+1)}
\end{align*}
$$
is an upper bound for the error on equidistant nodes in the infinity norm.

For Chebyshev nodes (8.7) we have that

$$
\begin{align*}
    \| f(x) - p_n(x) \|_\infty &\le \frac{\left( b-a \right) ^{n+1}}{2^{2n+1} \left( n+1 \right)!} M_{n+1} \\
    &\le \frac{1^{n+1}}{2^{2n+1} \left( n+1 \right)!} (2\pi)^{n+1} \\
    &= \frac{(2\pi)^{n+1}}{2^{2n+1} \left( n+1 \right)!} \\
    &=2\frac{\left( \pi / 2 \right) ^{n+1}}{(n+1)!}.
\end{align*}
$$

For the $2$-norm consider Lemma 8.1 which states that  

$$
\| g \|_2 \le W \| g \|_\infty \quad \text{for any function } g \in C[a,b].
$$  

Now set $g \equiv f(x) - p_n(x)$, then we get as $n \to \infty$ that $\| g \|_2 \to 0$ because $\| g \|_\infty \to 0$. The factorial will dominate in the Chebyshev case, and $n^{n+1}$ will dominate in the equidistant case. I.e., the bounds for $\| g \|_\infty$ goes to $0$ as $n \to \infty$ in both cases and in both norms.
$g$ is a linear combination of continuous functions, so the lemma holds.

For $f(x) = e^{3x}\left( \sin (2x) \right) $, remember that $\sin x = \Im (e^{ix})$. Then

$$
\begin{align*}
    f(x)&=e^{3x} \cdot  \Im (e^{2ix}) \\
    &= \Im (e^{(3+2i)x}).
\end{align*}
$$
So it is 
$$
\begin{align*}
    \left| f^{n+1}(x) \right| &= \left| (3+2i)^{n+1}\cdot  e^{3x}\left( \sin (2x) \right)\right|\\
    &\le \left| 3+2i \right|^{n+1} \cdot  \left| e^{3x} \right| \cdot \left| \sin (2x) \right| \\
    &\le \left( \sqrt{13} \right)^{n+1} \cdot \left| e^{3x} \right| 
\end{align*}
$$
and then 
$$
\begin{align*}
    M_{n+1} &\le \left( \sqrt{13} \right)^{n+1} \max_{x \in [0, \pi /4]} \left| e^{3x} \right|\\
    &= \left( \sqrt{13} \right)^{n+1} \cdot e^{3\pi /4}.
\end{align*}
$$

So a bound for $f(x) =e^{3x}\left( \sin (2x) \right) $ is then

$$
\begin{align*}
    \| f(x) - p_n(x) \|_\infty &\le \frac{M_{n+1}}{(n+1)!} \max_{x \in [a,b]} \left| \pi_{n+1}(x) \right| \\
    & \le e^{3\pi /4} \frac{\left( \sqrt{13} \right)^{n+1}}{(n+1)!}  \cdot \left( \frac{\pi /4-0}{n} \right)^{n+1}  \cdot \frac{n!}{4}\\
    &= e^{3\pi /4} \frac{\left( \sqrt{13} \pi \right)^{n+1}}{4^{n+2}\ n^{n+1}(n+1)}
\end{align*}
$$
for equidistant nodes. For Chebyshev nodes it is

$$
\begin{align*}
    \| f(x) - p_n(x) \|_\infty &\le \frac{M_{n+1}}{(n+1)!} \max_{x \in [a,b]} \left| \pi_{n+1}(x) \right| \\
    & \le e^{3\pi /4} \frac{\left( \sqrt{13} \right)^{n+1}}{(n+1)!}  \cdot \frac{\left( \pi /4-0 \right) ^{n+1}}{2^{2n+1}}\\
    &= e^{3\pi /4} \frac{\left( \sqrt{13} \pi /4 \right)^{n+1}}{2^{2n+1}(n+1)!}\\
    &= e^{3\pi /4} \frac{\left( \sqrt{13} \pi\right)^{n+1}}{2^{4n+3}(n+1)!}
\end{align*}
$$

The same argument as above holds for the limit and consequently the $2$-norm.
Consider also that both functions are entire analytic functions, so the Runge-phenomenon will not be a problem (pp. 187 Süli & Mayers).