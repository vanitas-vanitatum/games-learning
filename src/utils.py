

def hzcat(args, sep=''):
    arglines = [a.split('\n') for a in args]
    height = max(map(len, arglines))

    # Do vertical padding
    arglines = [lines + [''] * (height - len(lines)) for lines in arglines]
    # Initialize output
    all_lines = ['' for _ in range(height)]
    width = 0
    n_args = len(args)
    for sx, lines in enumerate(arglines):
        # Concatenate the new string
        for lx, line in enumerate(lines):
            all_lines[lx] += line
        # Find the new maximum horiztonal width
        width = max(width, max(map(len, all_lines)))
        if sx < n_args - 1:
            # Horizontal padding on all but last iter
            for lx, line in list(enumerate(all_lines)):
                residual = width - len(line)
                all_lines[lx] = line + (' ' * residual) + sep
            width += len(sep)
    # Clean up trailing whitespace
    all_lines = [line.rstrip(' ') for line in all_lines]
    ret = '\n'.join(all_lines)
    return ret
