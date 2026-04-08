# just a draft; do try to improve this later
# decrements a 32bit number and stores 1 in T2 cell if underflow occurred


# Layout:  C3; C2; C1; _C0_;  T1;  T2;
# meaning:      value      ; tmp vals;
# value: value to decrement;   0;   0;

>+<             # set zero flag (T1)
[>-<[->>+<<]]   # if C0 != 0 : reset zero flag; also transfer C0 to T2 regardless
>>[-<<+>>]<<    # transfer value back to C0
>[  # if zero flag is true :
    -<+<                        # reset T1; pre set zero flag in C0
    [>-<[->>+<<]]>>[-<<+>>]<<   # compute zero flag to C0

    >[  # if zero flag is true : decrement C2 in the same way
        -<+<[>-<[->>+<<]]>>[-<<+>>]<<

        >[  # if zero flag is true : decrement C3
            -<+<[>-<[->>+<<]]>>[-<<+>>]<<
            ->[->>>>+<<<<]>     # copy last zero flag into T2
        ]<

        ->>
    ]<

    ->>                         # decrement C1; select T1
]<
-               # decrement C0


# compact decrement:
>+<[>-<[->>+<<]]>>[-<<+>>]<<>[-<+<[>-<[->>+<<]]>>[-<<+>>]<<>[-<+<[>-<[->>+<<]]>
>[-<<+>>]<<>[-<+<[>-<[->>+<<]]>>[-<<+>>]<<->[->>>>+<<<<]>]<->>]<->>]<-

# compact decrement (no underflow reporting):
>+<[>-<[->>+<<]]>>[-<<+>>]<<>[-<+<[>-<[->>+<<]]>>[-<<+>>]<<>[-<+<[>-<[->>+<<]]>
>[-<<+>>]<<>[-<<->>]<->>]<->>]<-
