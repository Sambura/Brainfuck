# TOSTRING_U8 subprogram: convert a uint8 byte as to a string representation (decimal number)
# INPUT: x (byte)
# OUTPUT: resulting digits: "{D1}{D2}{D3}"; D1 and D2 could be null bytes

# expected input layout (8 bytes):    0; X;  0;  0;  0; 0; 0; 0
# output layout:                      0; D3; D2; D1; 0; 0; 0; 0

# get input (testing only)
>,<

>       # skip one null byte
>+      # set do_divide flag
[
    # division loop: Divide VAL by 10; store remainder in VAL; result in QNT
    [
        # loop layout:   VAL; do_divide; TEN; QNT; overflow_flag
        [-]                                 # reset do_divide;
        >++++++++++                         # initialize TEN to 10
        >+                                  # increment quotient
        <[<<[>>>>-<<<]>[<]<->>->>+<<]       # subtract TEN from VAL; set overflow_flag on overflow
        <+>>>[-<-<<-<++++++++++>>>>]<<<     # on overflow: decrement QNT; restore VAL; otherwise: set do_divide; select do_divide
    ]

    # layout: PVAL; VAL; 0; 0; 0
    >>[-<+<+>>]<< # move QNT to do_divide (new VAL) and TEN (new do_divide); select new VAL
    <+>         # put sentinel on PVAL
    >           # select new do_divide (has copy of new VAL; if nonzero: continue division)
]

<<[<]                           # reset cursor to cell 0
>>>>>++++++[-<++++++++>]<<<<<   # setup '0' character in cell 4

>                       # select first digit

# convert to ASCII
[
    -               # remove sentinel
    >>>             # select '0'
    [->+<<<<+>>>]   # add '0' to digit; also copy to one cell right
    <<              # select next digit
]
>>>[-]<<<           # reset '0' to null

<[<]    # reset cursor to cell 0

# output (testing only)
>[>]<[.<]

###########################################
###########################################
###########################################

# Compact subprogram form:
>>+[[[-]>++++++++++>+<[<<[>>>>-<<<]>[<]<->>->>+<<]<+>>>[-<-<<-<++++++++++>>>>]<<
<]>>[-<+<+>>]<<<+>>]<<[<]>>>>>++++++[-<++++++++>]<<<<[->>>[->+<<<<+>>>]<<]>>>[-]
<<<<[<]
