# MIN subprogram
# INPUT: x (byte); y (byte)
# OUTPUT: MIN(x; y) (byte)

# expected input layout:    X;         Y; 0; 0; 0
# output layout:            MIN(X; Y); 0; 0; 0; 0;

# get input (testing only)
,>,<

# copy X to cell 3
[->>+<<]>>[-<<+>>>+<]<<
# subtract X from Y; cell 4 for overflow
[->[>>>-<<]>[<]>>+<<<-<]
# on overflow: Y smaller than X; copy Y \sub X to cell 0; make sure Y cell is zeroed out regardless
>>>>[[-]<<<[-<+>]>>>]<<<[-]
# copy X from cell 3 to cell 0
>>[-<<<+>>>]<<<

# print output (testing only)
.


# Compact subprogram form:

[->>+<<]>>[-<<+>>>+<]<<[->[>>>-<<]>[<]>>+<<<-<]>>>>[[-]<<<[-<+>]>>>]<<<[-]>>[-<<<+>>>]<<<
