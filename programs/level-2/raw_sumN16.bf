# Sum N numbers (16 bit sum)
# input: N characters terminated with null byte
# output: sum of N characters as two characters

# layout: 1; 1; 0; X; C1; C2; overflow_flag
# where C1 is the first byte of the sum (least) and C2 is the second byte (greatest)

# init stop sequence
+>+>>

# get first number
,
# 1; 1; 0; _X_; 0; 0; 0

# start summing unless input is zero
[
    [ # add X to sum
        ->>>+               # set overflow flag
        <<+                 # increment C1
        [>>[-]<<<<]<<[>]    # reset overflow_flag
        >>>>[[-]<+>]        # increment C2 on overflow
        <<<                 # select X
    ]
    ,
]

# print result
>>.<.
