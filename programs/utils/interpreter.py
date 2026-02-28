from time import perf_counter

def _bf_execute_basic(program: str, in_data: list[int], iter_limit:int=50000000) -> list[int]:
    """ Execute brainfuck program.
        Simplest interpreter: works directly on a string with the program   """
    memory = [0] * 32768
    pc = 0
    ptr = 0
    iptr = 0
    stdout = []
    loop_stack = []
    loop_iters = 0 # count only loop iterations to save on performance

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
            loop_iters += 1
            if loop_iters >= iter_limit:
                raise Exception('Iteration limit reached')

        pc += 1
    
    return stdout

# TODO: replace in_data with input_generator and do something similar with output
def _bf_execute_preprocessed(program: str, in_data: list[int], iter_limit:int=100000000) -> list[int]:
    """ Execute brainfuck program.
        Processes program into list of opcodes before executing.
        Also combines consecutive +/- and </> instructions          """
    def preprocess(program: str):
        opcodes = [0] # 0 - NOOP, 1 - ADD, 2 - SHIFT, 3 - BEGIN_LOOP, 4 - END_LOOP, 5 - PRINT, 6 - GET
        consts = [0]
        loop_stack = []

        for i, c in enumerate(program):
            if c == '+' or c == '-':
                value = 1 if c == '+' else -1
                if opcodes[-1] == 1:
                    consts[-1] += value
                else:
                    opcodes.append(1)
                    consts.append(value)
            elif c == '<' or c == '>':
                value = 1 if c == '>' else -1
                if opcodes[-1] == 2:
                    consts[-1] += value
                else:
                    opcodes.append(2)
                    consts.append(value)
            elif c == '.':
                opcodes.append(5)
                consts.append(0) # we *could* count dots, but if would probably just hurt runtime performance, since '.' is almost never repeated
            elif c == ',':
                opcodes.append(6)
                consts.append(0) # same as with '.'
            elif c == '[':
                loop_stack.append(len(opcodes)) # PC + 1
                opcodes.append(3)
                consts.append(0)
            elif c == ']':
                if len(loop_stack) == 0:
                    raise Exception(f'Brainfuck syntax error: unexpected "]" at index {i}')

                loop_start = loop_stack.pop()
                past_loop_end = len(opcodes)
                opcodes.append(4)
                consts.append(loop_start - 1) # PC for BEGIN_LOOP
                consts[loop_start] = past_loop_end - 1 # PC for END_LOOP
        
        if len(loop_stack) > 0:
            raise Exception(f'Brainfuck syntax error: unmatched "["')

        return opcodes[1:], consts[1:]

    opcodes, consts = preprocess(program)
    memory = [0] * 32768
    pc = 0
    ptr = 0
    iptr = 0
    stdout = []
    loop_iters = 0

    while pc < len(opcodes):
        opcode = opcodes[pc]

        if opcode == 1: # ADD
            memory[ptr] = (memory[ptr] + consts[pc]) & 0xFF
        elif opcode == 2: # SHIFT
            ptr += consts[pc]
        elif opcode == 4 and memory[ptr] != 0: # END_LOOP
            pc = consts[pc]
            loop_iters += 1
            if loop_iters >= iter_limit:
                raise Exception('Iteration limit reached')
        elif opcode == 3 and memory[ptr] == 0: # BEGIN_LOOP
            pc = consts[pc]
        elif opcode == 5:   # PRINT
            stdout.append(memory[ptr])
        elif opcode == 6:   # GET
            if iptr < len(in_data):
                memory[ptr] = in_data[iptr] & 0xFF
                iptr += 1
            else:
                memory[ptr] = 0

        pc += 1

    return stdout

def bf_execute(program: str, in_data: list[int]=[], implementation='best') -> list[int]:
    "Execute a brainfuck program with a given input. Returns data printed to stdout by the program"
    if implementation == 'best' or implementation == 'preprocessed':
        return _bf_execute_preprocessed(program, in_data)
    elif implementation == 'basic':
        return _bf_execute_basic(program, in_data)
    else:
        raise Exception(f'Invalid implementation name: "{implementation}"')

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
