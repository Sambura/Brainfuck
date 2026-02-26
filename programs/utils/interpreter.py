from time import perf_counter

def _bf_execute_basic(program: str, in_data: list[int]) -> list[int]:
    memory = [0] * 32768
    iter_limit = 100000000
    pc = 0
    ptr = 0
    iptr = 0
    stdout = []
    loop_stack = []
    iters = 0

    while pc < len(program):
        c = program[pc]

        if c == '+':
            memory[ptr] = (memory[ptr] + 1) & 0xFF
        elif c == '-':
            memory[ptr] = (memory[ptr] - 1) & 0xFF
        elif c == '>':
            ptr += 1
        elif c == '<':
            ptr -= 1
        elif c == '.':
            stdout.append(memory[ptr])
        elif c == ',':
            if iptr < len(in_data):
                memory[ptr] = in_data[iptr] & 0xFF
                iptr += 1
            else:
                memory[ptr] = 0
        elif c == '[':
            if memory[ptr] != 0:
                if iters >= iter_limit: # just to not check it all the time, place it here instead of main loop
                    raise Exception('Iteration limit reached')
                loop_stack.append(pc)
            else:
                nest_level = 1
                while nest_level > 0:
                    cc = program[pc + 1]
                    if cc == '[':
                        nest_level += 1
                    elif cc == ']':
                        nest_level -= 1
                    pc += 1

        elif c == ']':
            pc = loop_stack.pop() - 1

        pc += 1
        iters += 1
    
    return stdout

def bf_execute(program: str, in_data: list[int]=[]) -> list[int]:
    "Execute a brainfuck program with a given input. Returns data printed to stdout by the program"
    return _bf_execute_basic(program, in_data)

if __name__ == '__main__':
    program = input("Input a brainfuck program: ")
    start_time = perf_counter()
    try:
        output = bf_execute(program)
    except Exception as e:
        print(f'Encountered error during execution: {e}')
        output = []

    print(f'Execution took {perf_counter() - start_time:0.2f}s')
    text = ''.join(map(chr, output))
    print(f'\nProgram output:\n{text}')
