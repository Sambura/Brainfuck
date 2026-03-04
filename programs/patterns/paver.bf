# initial layout: 0; 0; TV; _TC_; 0; 0; 0
# final layout:   0; TV; TV; TV; TV; ::: TV; TV; _0_

>  # leave first cell as 0
>+ # set target value (1)
>- # set cell count (255)

# start the paver
[
    -           # decrement cell count
    [->+<]      # transfer cell count right
    <[-<+>>+<]  # transfer target value and put one behind the paver
    >>          # select cell count
]

<[-]< # clean up

# Paver overhead: 29 instructions \plus target value and cell count setup
