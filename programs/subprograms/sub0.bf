# SUB0 subprogram
# SUMMARY: subtracts Y from X; clips result to 0 if Y is greater than X
# INPUT: y (byte); x (byte)
# OUTPUT: MAX(x \sub y; 0) (byte)

# expected input layout:    Y;    X; 0; 0; 0
# output layout:            SUB0; 0; 0; 0; 0;

# get input (testing only)
,>,<

# setup stop sequence
>>>+<<<
# subtract Y from X; keep track of overflow in cell 4
[->[>>>-<<]>[<]>>+<<<-<]
# on overflow: reset X to 0; clean up memory; move X to cell 0
>>>->[[-]<<<[+]>>>]<<<[-<+>]<

# output (testing only)
.

# Compact subprogram form:
>>>+<<<[->[>>>-<<]>[<]>>+<<<-<]>>>->[[-]<<<[+]>>>]<<<[-<+>]<
