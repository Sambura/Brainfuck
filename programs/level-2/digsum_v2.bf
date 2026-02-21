# Compute sum of two numbers and print the result
# Input: XY (two digit characters); output: ZZ (sum of the digits as a decimal number; only supports sums up to 19)

# _0_ 48 0 0
++++++[->++++++++<]

# 0 48 X _Y_
>>,>,

ASCII to numbers; select the second number
# 48 0 x _y_
<<[-<+>>->-<<]>>

Calculate and select the sum
# 48 0 _z_ 0
[-<+>]<

Setup tens counter
# 48 0 z _10_ 10 0
>>>+++++[-<++<++>>]<<

[
    <            # select sum
    [>>-<<<]<[>] # decrement overflow flag and reset to zero
    >->-         # decrement counters
]

# 48 0 z' _0_ overflow_flag 0
# overflow flag is set if the sum was less than ten

# invert overflow_flag and copy it elsewhere
# 48 0 z' overflow_flag _0_ !overflow_flag
>>+<[-<+>>-<]

# if !overflow_flag : print 1
>[[-]<<<<<+.->>>>>]<

# if overflow_flag : add 10 back to sum
# 48 0 z" _0_ 0 0
<[[-]<++++++++++>]

# print digit from sum cell
<[-<<+>>]<<.

