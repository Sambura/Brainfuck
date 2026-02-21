# Divide x by y; no ascii conversion
# input: xy
# output: quotient of x and y (not digits)

# layout: stop_one; stop_zero; X; Y; Y_tmp; nonzero_count; quotient;;

# init stop sequence
+>>

# get input
,>,
# 1; 0; X _Y_; 0; 0;

## division loop
[
    # subtract loop
    [
        <               # select X
        [>>>+<<<<]<[>]  # increment nonzero_count; reset to stop_zero
        >->->+          # subtract 1 from X and Y; add 1 to Y_tmp
        <               # reset to Y
    ]
    # 1; 0; X \ Y; _0_; Y; nonzero_count; 0;

    # increment quotient
    >>>+

    # restore Y; subtract Y from nonzero_count
    <<[<+>->-<]
    # 1; 0; X'; Y; _0_; overflow_flag; quotient?;

    # on overflow : decrement quotient and reset Y to zero to exit the loop
    >[[-]>-<<<[-]>>]
    # 1; 0; X'; Y; 0; _0_; quotient;

    # reset to Y
    <<
]
# 1; 0; X'; _0_; 0; 0; quotient;

# print quotient
>>>.
