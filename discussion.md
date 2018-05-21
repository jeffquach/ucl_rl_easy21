# Discussion

## What are the pros and cons of bootstrapping in Easy21?

Pros: 
    - allows for faster convergence than non-bootstraping techniques b/c you are sampling in direction of action-value function to perform update
    - less likely to lead to a result with high variance and can learn faster as a result

Cons:
    - more bias can be introduced because of the nature of sampling from action-value matrix to perform updates and the starting action-values and policy values

## Would you expect bootstrapping to help more in blackjack or Easy21? Why?

Boostrapping would likely help more in Easy21 since there is more predictability given that there are less cards and the rules are less complicated.  The result of this is the game is blackjack will have more state spaces making bootstrapping less effective.

## What are the pros and cons of function approximation in Easy21?

Pros:
    - for environments/scenarios where the state space is large it prevents you from maintaining this, thus resulting in lower memory usage and prevention of exploring all state spaces
    - able to generalize seen states to unseen states
    - reduces learning time

Cons:
    - SGD update may be inefficient to converge to an optimal solution
    - some imprecision may be introduced since generalizations are being made compared to table lookups in MDPs

## How would you modify the function approximator suggested in this section to get better results in Easy21?

Use batch methods to speed up convergence time