# Loading bar animation
# Animates a progress bar that looks something like this:
# / Please wait   (                                                 ################           ) 3
# Features:
#   * 4 character spinner (/|\ and minus character)
#   * bar that moves left to right
#   * 8 bit counter after the bar that counts up every time bar completes a "revolution"

# Memory layout
# 0; 0; 47;  124;  92;  45; 91; 13;  0;  32; 35; FC; BC; SC; 0; 0; 60; 7; OF;  0;   0;   0;  0;    0;   0;   0;
# 0; 0; '/'; '|'; '\'; MIS; OB; \r; SS; ' ';'#'; __________; tmp;  TS; BS; _; tmp2;tmp3;tmp4;tmp5;tmp6;tmp7;CNT
# CONST SECTION =========================)

# FC: first count (spaces); BC: bar count (hashes); SC: second count(spaces); OF: bar offset; TS: total bar size; BS: bar size; SS: stop sequence

# ==================================
# ==================================
# ==================================

# init memory
++++[->++++++<]>-[>++>+++++>++++>++>++++[<]>-]>+>+++++++++>>->-> # all cells before CR
+++++++++++++                   # CR; remove 3 pluses to switch to LF (for debug)
>++++++++[->++++>++++<<]>>+++   # ' ' and '#'
>>>>>++++++++++++++[->++++++++>++<<]>++++               # TS and BS

# increment CNT
>>>>>>>>>+<<<<<<<<<

# select \r
<<<<<<<<<

# MAIN LOOP
[
    .           # print carriage return

    <<.             # print spinner
    [-<<<<+>>>>]    # move spinner character to the end of the list
    <               # select beginning of the list
    [[->+<]<]       # move list 1 cell right
    >>[>]           # reset to SS null byte

    >.              # print space

    >>>>>>>>>>      # select tmp2
    # print 'Please wait(ellipsis)   ' (only uses 5 cells)
    ++++[->++++<]>[->+++++>+++++++>++<<<]>.>----.<+++++++++++++++++++++.----.>++++++
    +.<++++.>>.<++++.<----.++++++++.>---.>++++++++++++++...--------------...[[-]<]<
    <<<<<<<<<<<<    # back to CR

    <.              # print opening bracket
    >>>>>           # select FC
    [-<<.>>]        # print spaces
    >               # select BC
    [-<<.>>]        # print hashes
    >               # select SC
    [-<<<<.>>>>]    # print spaces
    <<<<<<<         # select OB
    ++.--           # print closing bracket
    >>>.            # print space

    >>>>>>>>>>>>>>>>    # Select CNT
    [->+<]>[-<+>>+<]    # copy CNT two cells forward
    >>+[[[-]>++++++++++>+<[<<[>>>>-<<<]>[<]<->>->>+<<]<+>>>[-<-<<-<++++++++++>>>>]<<
    <]>>[-<+<+>>]<<<+>>]<<[<]>>>>>++++++[-<++++++++>]<<<<[->>>[->+<<<<+>>>]<<]>>>[-]
    <<<<[<]  # subprograms/tostring_u8
    >[>]<[.[-]<]        # print converted CNT
    <<<<<<<<<<<<<<<<<   # select ' '
    ..                  # print two spaces

    >>>>>>>>>+                      # increment offset

    # populate FC = min(max(0; offset \sub BS); TS)
    [->+<]>[-<+<<<<<<<+>>>>>>>>]    # copy offset to FC
    <<[-<<+>>]<<[->>+<<<<+<+>>>]    # copy BS to BC and SC
    <<<                             # select BC
    [-<[>>-<<<<<]<<<[>]>>>->]       # subtract BC from FC; SC as overflow flag
    >[[-]<<[+]>>]                   # on overflow: reset FC and SC
    <<[->>>>>>>>+<<<<<<<<]          # move FC to tmp2
    >>>>>[-<+>]<[->+>>>>+<<<<<]>>>> # copy TS to tmp3; tmp2 is selected
    [->>+<<]>>[-<<+>>>+<]<<[->[>>>-<<]>[<]>>+<<<-<]>>>>[[-]<<<[-<+>]>>>]<<<[-]>>[-<<<+>>>]<<<   # subprograms/min
    [-<<<<<<<<+>>>>>>>>]            # move MIN(TS; FC) to FC

    # populate BC = min(BS; offset; max(0; TS \add BS \sub offset))
    <<<                             # select TS
    [->>>+<<<]>>>[->+<<<<+>>>]      # copy TS to tmp3
    <<[->>+<<]>>[-<<+>>>+<]         # add BS to tmp3
    <[->>>+<<<]>>>[-<<+<+>>>]<<     # copy offset to tmp2; select tmp2
    >>>+<<<[->[>>>-<<]>[<]>>+<<<-<]>>>->[[-]<<<[+]>>>]<<<[-<+>]< # subprograms/sub0
    [->>+<<]                        # move result to tmp4
    <[->+<]>[-<+>>+<]>              # copy offset to tmp3; select tmp3
    [->>+<<]>>[-<<+>>>+<]<<[->[>>>-<<]>[<]>>+<<<-<]>>>>[[-]<<<[-<+>]>>>]<<<[-]>>[-<<<+>>>]<<<   # subprograms/min
    <<<                             # select BS
    [->>+<<]>>[-<<+>>>>+<<]>        # copy BS to tmp4; select tmp3
    [->>+<<]>>[-<<+>>>+<]<<[->[>>>-<<]>[<]>>+<<<-<]>>>>[[-]<<<[-<+>]>>>]<<<[-]>>[-<<<+>>>]<<<   # subprograms/min
    [-<<<<<<<<+>>>>>>>>]            # copy result to BC; tmp3 is selected

    # populate SC = TS \sub FC \sub BC
    <<<<                        # select TS
    [-<<+>>]<<[->+>+<<]         # copy TS to second@tmp
    <<                          # select BC
    [->>+<<]>>[->-<<<+>>]       # subtract BC from second@tmp
    <<<                         # select FC
    [->>>+<<<]>>>[->-<<<<+>>>]  # subtract BC from second@tmp
    >[-<<+>>]                   # copy second@tmp to SC (just compute in SC next time)

    # reset offset if offset is greater than TS \add BS
    >[-<+>]<[-<+>>+<]               # copy TS to first@tmp
    >>[-<<+>>]<<[-<+>>>+<<]         # add BS to first@tmp
    >>>[-<<<+<->>>>]<<<[->>>+<<<]   # subtract offset from first@tmp
    <+++++>                         # increment first@tmp by 5 
    +<[[-]>-<]>                     # compute `offset == TS \add BS` to second@tmp; select it
    [[-]>>>[-]>>>>>>>+<<<<<<<<<<]   # reset offset if true; also increment CNT

    <<<<<<<     # select stop sequence null

    # waste some time to slow down the animation :)
    -[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]
    -[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]
    -[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]
    -[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]
    -[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]
    -[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]
    -[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]
    -[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]
    -[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]
    -[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]
    -[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]
    -[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]-[-]

    <        # select CR
]
