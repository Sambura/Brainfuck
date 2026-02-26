# N glider's goal is to travel N cells in a direction while using few instructions; this one only works if all cells are zeros
# input: N
# output: None

# set travel distance; eg 100 to the right:
>++++++++++[-<++++++++++>]<

# start gliding
[
    [->+<]  # transfer travel count to next cell
    >       # travel 1 cell right
    -       # decrement travel amount
]

# examples:
# 
# zero travel
#   starting layout: _0_; 0; 0; 0; 0
#   ending layout: _0_; 0; 0; 0; 0
# 
# singular travel
#   starting layout: _1_; 0; 0; 0; 0
#   ending layout: 0; _0_; 0; 0; 0
#
# long travel
#   starting layout: _4_; 0; 0; 0; 0
#   ending layout: 0; 0; 0; 0; _0_

# free bonus: 255 glider
-[
    [->+<]  # transfer travel count to next cell
    >       # travel 1 cell right
    -       # decrement travel amount
]

# free bonus x2: 510 glider
-[
    [->>+<<]    # transfer travel count to next cell
    >>          # travel 1 cell right
    -           # decrement travel amount
]

# presumably you could also do some smart stuff with decrementing (or incrementing) a custom amount
# to get more efficient gliders (setup cost vs glider cost); but idk