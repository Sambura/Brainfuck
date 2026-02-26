# Parse a decimal input number into an internal representation (a byte)
# input: 1 to 3 digits
# output: corresponding ASCII character

# layout: 1; 0; X; result; tmp1; '0'; tmp2;

# init stop sequence
+>>

# init ASCII conversion ('0' = 48)
>>++++++[->++++++++<]<<<
# 1; 0; _0_; 0; 0; '0'; 0;

# get first digit
,

# start parsing
[
    >>>[->+<]>          # put '0' to tmp2
    [-<+<<<->>>>]       # reset '0' slot; convert input to number
    <<<[->++++++++++<]  # set tmp1 to result x 10
    >[-<+>]<<[->+<]     # compute new result
    ,                   # get next digit
]

# print result
>.
