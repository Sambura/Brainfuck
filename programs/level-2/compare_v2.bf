# Compare input character to K (75) : L if less or equal; G if more

71/76
+++++++++[->++++++++>++++++++<<]
>->++++>>>,

>>+++++[-<+++++++++++++++>]<

# setup complete; memory state:
# 0 G L 0 0 ? _K_ 0 0 

[
    -<-     Decrement both; look at input
    [<]>    Move right once if input reached zero
    >       Return to counter
]           Stop if counter reached zero or if we are off counter now

Print the answer
<<<<<.
