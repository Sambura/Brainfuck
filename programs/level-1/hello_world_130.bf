# Print "Hello(comma) world!" to console
# Relatively simple implementation; a bit of optimizations (130 instructions)

# ASCII values:
# H:        0x48 (72)
# e:        0x65 (101)
# l:        0x6C (108)
# l:        0x6C (108)
# o:        0x6F (111)
# (comma):  0x2C (44)
# (space):  0x20 (32)
# w:        0x77 (119)
# o:        0x6F (111)
# r:        0x72 (114)
# l:        0x6C (108)
# d:        0x64 (100)
# !:        0x21 (33)

# ============================================
# ============  CODE STARTS HERE  ============
# ============================================

store 8 to first cell
++++++++

inc second cell by 9 and third by 4 | 8 times stop head at second cell
[>+++++++++>++++<<-]>

print H
.

add 32 from second cell and put 32 in fourth
>[<+>->+<]<

print e
---.

print ll
+++++++..

print o
+++.

print comma
>>++++++++++++.

print space
------------.

print w
<<++++++++.

print o
--------.

print r
+++.

print l
------.

print d
--------.

print !
>>+.
