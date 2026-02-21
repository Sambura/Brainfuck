# Compare first input character to second : L if less; G if more; E if equal

69/76/71
+++++++++[->++++++++>++++++++>++++++++<<<]
>--->++++>->>>,>>,

# setup complete; memory state (I is input; K is second input (referred to as counter)):
# 0 E L G 0 0 I 0 _K_ 0 0 0 0 0

[
    -<<-      Decrement both; look at input
    [>>[<]]<  Land in different spots depending on current input and counter values
    >>
]

# if input reached zero (counter unknown)
# 0 E L G 0 0 0  _0_ K? 0   0 0 0 0 0

# if input is nonzero but counter is zero
# 0 E L G 0 0 I   0  0 _0_  0 0 0 0 0

# move right and check counter value
>[>]

# if input reached zero (counter zero)
# 0 E L G 0 0 0 0 _0_  0 0 0 0 0 0
# equal

# if input reached zero (counter nonzero)
# 0 E L G 0 0 0 0  K? _0_ 0 0 0 0 0
# less

# if input is nonzero but counter is zero
# 0 E L G 0 0 I 0  0   0 _0_ 0 0 0 0
# greater

<<<<<<<.
