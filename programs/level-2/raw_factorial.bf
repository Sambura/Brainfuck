# Calculate raw factorial of input (no ASCII conversion) (8 bit)
# input: x
# output: x!

# layout: X; mult_f1; mult_f2; mult_tmp; result_tmp;

# get input
,
# _X_

# init result
>>>>+<<<<
# _X_; 0; 0; 1; 

# factorial loop
[
# copy factors to mult_f1 and mult_f2
[->>+<<]>>[-<+<+>>]>>
# X; X; 0; 0; _result_; 0
[-<<+>>]<<<
# X; _X_; result; 0; 0; 0

# run multiplication
[
    -            # decrement mult_f1 
    >[->+<]>     # move mult_f2 to mult_tmp
    [<+>->+<]    # add mult_f2 to result; restore mult_f2
    <<           # select mult_f1
]

# reset factors; decrement input
>[-]<<-
# _X \ 1_; 0; 0; 0; result'; 0
]

# select and print result
>>>>.
