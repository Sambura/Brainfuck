# U32_INC subprogram: increment a 32bit unsigned int (no overflow detection)
# INPUT: V3; V2; V1; V0 (4 bytes): a number to increment (MSB to LSB)
# OUTPUT: V3'; V2'; V1'; V0' (same 4 bytes): the input value incremented by 1


# Layout:  V3; V2; V1; _V0_;  0;  0
# meaning:   input value   ; T0; T1

+>+<        # increment V0; set overflow flag (T0)
[>-]>[>]<   # if V0 != 0 : reset overflow flag; select T0

# if overflow:
[
    -<+<+   # reset overflow flag; set new overflow flag (V0) and increment V1
    [>-]>[>]<[-<+<+[>-]>[>]<[-<<+>>]>]> # same logic as above 
]<


###########################################
###########################################
###########################################

Compact subprogram form:
+>+<[>-]>[>]<[-<+<+[>-]>[>]<[-<+<+[>-]>[>]<[-<<+>>]>]>]<
