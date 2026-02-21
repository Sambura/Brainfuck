# Print ASCII code for a character
# input: x (a character)
# output: ord(x) (x's ASCII code) 

# init stop sequence
# _1_; 0; 0; 0; ///
+

# Take input in cell 2
# 1; 0; _X_; 0; ///
>>,

### cell layout: stop_one; stop_zero; X_tmp_1; X_tmp_2; overflow_flag; order_counter_tmp1; order_counter_tmp2; unassigned_tmp; unassigned_tmp2; xx; hundreds; tens; ones

## calculate hundreds

# put 100 in cell 7
# 1; 0; X; 0; 0; 0; _0_; 100; 0; 0; XXXXX
>>>>++++++++++[->++++++++++<]

# copy 100 to cells 4 5 and 6; select order_counter_tmp1
# 1; 0; X; 0; 100; _100_; 100; 0; 0; XXXXX
>[-<+<+<+>>>]<<

# magic line that reset counters back to zero if the input is zero
# same logic as edge case handler below; slightly different offsets; doesn't shift cursor
<<<[>]<<[>>>>[[-]>]>[-]<<<<<<<]>>>>

[   ## start counting hundreds

# subtract 100 from X
# 1; 0; X \ 100; 0; overflow_flag; _0_; 100; 0; 0; XXXXX
[
    -<<<-       # subtract 1 from counter and from input
    [>>-<<<]<   # subtract 1 from flag counter unless input is zero; land on either stop_one or stop_zero
    [>]         # (reset) select stop_zero
    >>>>        # select order_counter_tmp1
]

# increment hundreds
# 1; 0; X'; 0; overflow_flag; 0; 100; 0; 0; 0 | _H@1_; 0; 0
>>>>>+

# copy overflow flag to unassigned_tmp2
# 1; 0; X'; 0; _0_; 0; 100; 0; overflow_flag; 0 | H; 0; 0
<<<<<<[->>>>+<<<<]

# reset counters
# 1; 0; X'; 0; 100; 100; 100; _0_; overflow_flag; 0 | H; 0; 0
>>[->+<]>
[-<+<+<+>>>]

# edge case: if overflow_flag is set but X is at zero : reset overflow_flag
# 1; 0; X'; 0; _0_; 0; 100; 0; overflow_flag?; 0 | H; 0; 0
<<<<<[>]<< # select 1 if X is zero; select 0 otherwise
[>>>>[[-]>]>[-]<<<<<<<] # if 1 is selected : reset overflow_flag and counters to zero
>>>>>>                  # reset position to unassigned_tmp1

# check overflow flag : if nonzero decrement hundreds and exit loop
>       # select unassigned_tmp2 (overflow_flag)
[
    [-]     # reset unassigned_tmp2 (overflow_flag)
    >>-     # decrement hundreds counter
    <<<<<<  # select overflow_flag
    [-<<+>>]# transmit 100 from overflow_flag to X
    >[[-]>] # reset order_counter_tmp1 and order_counter_tmp2
    >       # select unassigned_tmp2
]
<<<     # select order_counter_tmp1

]   ## stop counting hundreds
# 1; 0; X'; 0; 0; _0_; 0; 0; 0; 0 | H; 0; 0


## count tens now

>++[->+++++<]>[-<+<+<+>>>]<<        # setup counters
<<<[>]<<[>>>>[[-]>]>[-]<<<<<<<]>>>> # reset counters if we already at zero

[ # counter loop
    [-<<<-[>>-<<<]<[>]>>>>] # subtract loop
    >>>>>>+ # increment tens
    <<<<<<<[->>>>+<<<<]>>[->+<]>[-<+<+<+>>>] # cleanup
    <<<<<[>]<<[>>>>[[-]>]>[-]<<<<<<<]>>>>>>  # handle zero edge case
    >[[-]>>>-<<<<<<<[-<<+>>]>[[-]>]>]<<< # revert iteration on overflow
]

# 1; 0; O; 0; 0; _0_; 0; 0; 0; 0 | H; T; 0

# transmit ones
# 1; 0; 0; 0; 0; 0; 0; 1; 0; 0 | _H_; T; O; 0; 1; 1; (second stop sequence)
<<<[->>>>>>>>>>+<<<<<<<<<<]>>>>>>>>>>>>+>+<<<<<

# preprocess for output
# 1; 0; 0; 0; 0; h; t; 0; _0_; 0 | H; T; O; 0; 1; 1; 
# where h is : print hundreds? and t is : print tens?
[<<<<<+>+>>>]>>>>[<]<< # hundreds
[<<<<<+>>>]>>>>[<]<<<<< # tens

# generate zero (48)
# 1; 0; 0; 0; 0; h; t; 0; _0_; '0' | _H_; T; O
++++++++[->++++++<]

# add zero to all digits
>[->+>+>+<<<]<<<<

# print hundreds (if nonzero)
[[-]>>>>>.<<<<<]

# print tens (if nonzero)
>[[-]>>>>>.<<<<<]

# print ones
>>>>>>.
