# Sum N numbers (16 bit sum) (with ASCII conversion)
# input: N characters terminated with null byte
# output: sum of N characters as two characters

# print prompt "Enter a space separated list of numbers to get their sum: "
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>++++++++++++++++[-<++
<++++<+++++++<+++++++<+++++++<++<+++++++<+++++++<++++++<++++++<+++++++<++<++++++
+<++++++<++++++<++<+++++++<+++++++<++<+++++++<+++++++<++++++<++++++<+++++++<++++
+++<+++++++<++<++++++<+++++++<++<+++++++<+++++++<+++++++<+++++++<++<++++++<+++++
+<+++++++<++++++<+++++++<++++++<+++++++<++++++<+++++++<++<++++++<++++++<++++++<+
++++++<+++++++<++<++++++<++<+++++++<++++++<+++++++<+++++++<++++>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<------<---<+++++<+++<<++<-------<+++
++<++++++++<++++<<++++<+++++<+++++++<<-<++++<<+++<++<+++++<++<---<+++++<--<<++++
++<-<<++++<+++<-------<----<<++++<+++++<++++<+<++<+<<+++++<+++<<+++++<+++<+<<+++
<<+<<++<+++++<++++<--<+++++>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>[<]>[.>]<[[-]<]

# struct Int8: { value; tmp }
# struct Int16: { least_byte; greatest_byte; tmp }
# SS: stop sequence

# LAYOUT:       SS  |  Int16  |Byte|   SS   |  Int16(rev) | Int8 |  Byte  |   Int8 & Byte  |  Byte  | Byte
# LAYOUT:       SS;   Result;  tmp;    SS;   CurrentValue; Ten   Input;  Space; ZeroOffset;  Carry;  ErrorChar
#        1  1  1  0   #  #  #   #    1   0      #  #  #    10 0    #      ' '   #    16        #        #

# init stop sequences
+>+>+>>>>>>+>>

# init Ten
>>>++++++++++

# get first digit from input
>>,

# init Space to ' ' and ZeroOffset to 16
>>++++[-<++++++++>>++++<]<<

# start summing unless input is null byte
[
    ### parse input digit
    >[->+<]>[-<+<->>]<<         # subtract Space from Input; _Input_ selected
    >>+<<                       # set space flag (tmp@Space)
    [   # if _Input_ was not space;
        >>[-]<<                 # reset space flag (tmp@Space)
        >>>[-<+>]<[-<<->>>+<]<< # subtract ZeroOffset from _Input_

        # check if input character is actually a digit
        <<[->+>>>+<<<<]>                # transfer value@Ten to tmp@Ten and tmp@Space; select tmp@Ten
        [-<+>>[>>->>]>[>]<[>]<<<<-<]    # subtract tmp@Ten from Input and keep track of overflow in tmp@Space
        >++++++++++                     # select _Input_ and restore value
        >>>>+<<[->>-<<]                 # invert tmp@Space and put it in Carry
        >>[[-]                          # if Input was greater or equal to 10; set error char
            <<<<[->+<]>[->>+<<]+>>[->>+<<]+ # put error character in ErrorChar; Space and ZeroOffset are reset to one
            >                               # select Carry
        ]
        <<<<                            # select Input

        # (handle for least_byte@Current being zero)
        <<<<<++                             # set null flag (tmp@Current)
        >>[<<--]<<-[<<-]+>>                 # reset null flag
        [[-]>[->->>>>>>>+<<<<<<<<<]<<[>]>]  # transfer 256 from greatest_byte@Current to least_byte@Current (and set Carry)
        >>                                  # select least_byte@Current
        [ # multiply Current by Ten and put it into tmp as (LB) and tmp@Result as (GB)
            [ # while (least_byte@Current != 0)
                -                                               # decrement least_byte@Current
                >>>>>>>[-<<<<<<<+>>>>>>>]<<<<<<<                # restore value from Carry
                >[->+<]>                                        # put value@Ten to tmp@Ten
                [-<+<<<+<<<+[>>>-<]>[>]<[>]>[[-]<<<<+>>>>]>>>>] # iterate multiplication
                <<                                              # select least_byte@Current
            ]
            <                                               # select greatest_byte@Current
            [->->>>>>>>+<<<<<<<<<]<<[>]>>>                  # transfer 256 from greatest_byte@Current to least_byte@Current (and set Carry)
        ]
        <<<<<[->>>>>+<<<<<]<[->>>>>+<<<<<]          # restore Current
        >>>>>>>>>                                   # select Input
        [-<<<<<++>>+[<<--]<<-[<<-]+>>[[-]>+<]>>>>>] # add Input to Current
    ]   # otherwise fallthrough and exit parse loop

    ### on space or null byte : sum the result and the current numbers
    ,>>+                # get new input digit; select and increment space flag (tmp@Space)
    <<[>>-<<<]<[>]      # if input is not zero decrement space flag
    >>>                 # select space flag (tmp@Space)
    [   # if space flag is set (tmp@Space); add Current to Result
        [-]<<<<<<                                   # reset space flag and select greatest_byte@Current
        [-<<<<<<+>>>>>>]                            # add greatest bytes together
        >[-<<<<<<+<<+[>>-]<<<[>]>>>[[-]<+>]>>>>>>]  # add least_byte@Current to Result
        >>>>>                                       # select tmp@Space
    ]

    ### check error char and exit loop if set
    >>>[<<<<<[-]>>>>]<[>]

    <<<<    # select _Input_
]


######
######  Compute section end; Converting result to string now
######


+       # set Input (meaning: display_result_flag)
## check ErrorChar and print error if set
>>>>>[ # Error occurred; print error message and offending character
    >           # deselect ErrorChar
    # print "\nFailed to compute sum: invalid character: "
    >++++++++++>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>+++++++++++++++++++++++++++++++++++
    [-<+++<+++<+<+++<+++<+++<+++<+++<+++<+++<+<++<+++<+++<+++<+<+++<+++<+++<+++<+++<
    +++<+++<+<+++<+++<+<+++<+++<+++<+++<+++<++>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<-<-
    -----<---<-----<<+++<--------<+++++++++++++<+++++<<---<------------<++++<+++++++
    +++++<++++++++++<---<----<+++++++++++<++++++++++++<+++++++<++++<++++++<------<--
    -<++++++<+++++++++++<---<-----<----<+++<<-------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    >>>>>>>>>>+++++++++++++++++++[-<++<+++<++++++<+++++<++++++<+++++<+++++<++++++<++
    +++>>>>>>>>>]<------<+<<++++++<++<++++<++<<++>>>>>>>>[<]>[.>]<[<]
    <.[-]       # select and print ErrorChar
    <<<<<[-]    # select and reset Input (display_result_flag)
    >>>>>       # select ErrorChar to align cursor
]<<<<<

[   # if display_result_flag is set : print the result
    [-]
    >>>[-<<+>>]<<                               # Make '0' from Space and ZeroOffset (put in Space)
    [-<<<<+>>>>]<<<[-<<+>>]<<<<<-               # mutate memory layout:

    # LAYOUT:       SS; Result; tmp0; tmp1; overflow_flag; do_div; ten_tmp; Ten; Zero; spacer; Digit0; Digit1; Digit2; etc
    #        1  1  1  0  LB GB    0     0        _0_          0       0      10   ' '     0      0        0      0

    # print "\nResult: "
    >>>>>>++++++++++>>>>>>>>>+++++++++++++++++++++++++++++[-<+<++<++++<++++<++++<+++
    +<+++<+++>>>>>>>>]<+++<<<--------<+<-<++++++++++++++<----->>>>>>>[<]>[.>]<[[-]<]
    <<<<<

    >+  # set do_div
    [   # digit string computation loop
        [   # division loop
            [-]                                     # reset do_div
            >>[-<+<<+>>>]<                          # put Ten in ten_tmp and overflow_flag; select ten_tmp
            [->+<<<<<<<[>>>>-<<<<<]<[>]>->>>>>>]    # subtract ten from LB; keep track of overflow
            +<<<<<[>>>>>-<<<<<<<]<<[>]>>>>>         # compute is_gb_zero into ten_tmp; select overflow_flag
            [->+<]>[-<+>>+<]<                       # add overflow_flag to ten_tmp; keep overflow_flag value
            [[-]<<<->>>]                            # on overflow: decrement GB; clear overflow_flag
            >++>[-<->]+<[[-]>-<]>                   # select ten_tmp; ten_tmp = ten_tmp == 2; (stop_condition)
            <+>                                     # set do_div; select ten_tmp
            [[-]<-<<<<<++++++++++>>>>>>]            # on stop_condition: reset do_div; increment LB by 10
            <[->+<]>[-<+<+>>]<<                     # copy do_div into overflow_flag; select overflow_flag
            [   # on overflow_flag: increment division result
                # reuse set overflow_flag for tmp0 increment:
                <<+                 # increment tmp0
                [>>-<<<<<]<<<[>]    # reset overflow_flag
                >>>>>[[-]<+>]       # increment tmp1
            ]
            >                                       # select do_div
        ]
        >>>>>[>]<                               # select Last Digit* cell with nonzero value
        [[->+<]<]                               # shift all Digit*'s by 1 to the right
        <[->+>+<<]>[-<+>]                       # put '0' into Digit0
        <<<<<<<<[+]<[->>>>>>>>>>+<<<<<<<<<<]    # add LB to Digit0 and reset GB to zero
        >>[->>+<<<<+>>]>>[[-]>+<]               # put tmp0 to LB; set: do_div = LB != 0
        <[->+<<<+>>]>[[-]>+<]                   # put tmp1 to GB; increment do_div if GB != 0
        >                                       # select do_div
    ]

    # print all digits in order now
    >>>>>[.>]
]

