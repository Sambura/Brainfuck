# rawmult
# input: xy
# output: product of x and y ascii values as an ascii character

input
,>,<

repeat #firstcell times
[->
    add second cell to third and fourth cell
    [->+>+<<]>
    put value from third cell back into second
    [-<+>]
<<]

print the fourth cell
>>>.
