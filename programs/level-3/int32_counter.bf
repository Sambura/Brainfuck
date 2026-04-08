# count up indefinitely (32 bit)
# print the decimal counter to console

# static init
+

# cell#      00; 01; 02; 03; 04; 05;   06;     07; 08; 09; 10; 11; 12;
# Layout:     1;  0; B3; B2; B1; B0;   T0;      0; C3; C2; C1; C0;
# meaning: stop seq;   int32 bytes ;  tmp; anchor;   int32 copy  ;

# while True (use SS to loop forever)
[
    $% increment timing
    >>>>> # select B0

    # 32bit increment
    +>+<[>-]>[>]<[-<+<+[>-]>[>]<[-<+<+[>-]>[>]<[-<<+>>]>]>]<
    
    [->+<]>[-<+>>>>>>+<<<<<]<               # copy B0 to C0
    <[->>+<<]>>[-<<+>>>>>>+<<<<]<<          # copy B1 to C1
    <[->>>+<<<]>>>[-<<<+>>>>>>+<<<]<<<      # copy B2 to C2
    <[->>>>+<<<<]>>>>[-<<<<+>>>>>>+<<]<<<<  # copy B3 to C3
    
    >>>>>>>>>  # select C0

    # Layout:  ////////////; C3; C2; C1; C0; T1; T2;  10;  0; Q3; Q2; Q1; Q0;  0;  0; D0; D1; D2; D3; etc
    # meaning: ////////////;   value       ;  tmps ; TEN; T0;    quotient   ; T3; T4;   digits      ; etc

    >+      # set T1 (do_divide)
    # digitizer loop
    [
        # division loop
        [
            ->>         # reset do_divide; select TEN
            ++++++++++  # initialize TEN to 10
            >>>>>       # select Q0
            +>+<[>-]>[>]<[-<+<+[>-]>[>]<[-<+<+[>-]>[>]<[-<<+>>]>]>]<    # 32bit increment
            <<<<<       # select TEN

            # subtraction loop : subtract 10 from value; store nonzero value in T0 if value hit zero during subtraction
            [
                -                   # decrement TEN
                <<<                 # select C0

                # 32bit decrement; if value is zero : sets T2
                >+<[>-<[->>+<<]]>>[-<<+>>]<<>[-<+<[>-<[->>+<<]]>>[-<<+>>]<<>[-<+<[>-<[->>+<<]]>
                >[-<<+>>]<<>[-<+<[>-<[->>+<<]]>>[-<<+>>]<<->[->>>>+<<<<]>]<->>]<->>]<-

                >>[->>+<<]>         # transfer T2 to T0; select TEN
            ]

            <<+     # set do_divide

            >>>     # select T0
            [   # on overflow during subtraction
                [-]                             # reset T0
                <<<-                            # reset do_divide
                <++++++++++<[+]<[+]<[+]         # restore value
                >>>>>>>>>>>                     # select Q0

                # 32bit decrement (no overflow version)
                >+<[>-<[->>+<<]]>>[-<<+>>]<<>[-<+<[>-<[->>+<<]]>>[-<<+>>]<<>[-<+<[>-<[->>+<<]]>
                >[-<<+>>]<<>[-<<->>]<->>]<->>]<-

                <<<<                            # back to T0
            ]

            <<<     # select do_divide
        ]

        # generate the next digit
        >>>>>>>>>>                  # select D0
        [>]<                        # select last nonzero D# (or T4 if no digits are there yet)
        [[->+<]<]                   # shift all digits one to the right; end up on T4
        ++++++[->++++++++<]         # initialize D0 to '0'
        <<<<<<<<<<                  # select C0
        [->>>>>>>>>>>+<<<<<<<<<<<]  # add C0 to D0

        # copy quotient back to value
        >>>>>>>>                            # select Q0
        [>+<[-<<<<<<<<+>>>>>>>>]]<          # copy Q0; increment T3 if nonzero
        [>>+<<[-<<<<<<<<+>>>>>>>>]]<        # copy Q1; increment T3 if nonzero
        [>>>+<<<[-<<<<<<<<+>>>>>>>>]]<      # copy Q2; increment T3 if nonzero
        [>>>>+<<<<[-<<<<<<<<+>>>>>>>>]]     # copy Q3; increment T3 if nonzero

        >>>>[[-]<<<<<<<<+>>>>>>>>]          # set do_divide if T3 is nonzero
        <<<<<<<<                            # select do_divide
    ]

    >>>>>>>>>>                      # select D0
    <++++[-<++++++++>]<....[-]>>    # do indent
    [.>]+++++++++++++.              # print the number and \r
    [[-]<]                          # reset the memory
    <<<<<<<<<<<<<<<<                # select B0

    <<<<<   # select cell #0 (SS)
]
