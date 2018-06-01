import tensorflow as tf
import uuid

from pathlib import Path


class logger:
    path = Path('logdir')
    writer = tf.summary.FileWriter(str(path / str(uuid.uuid4())))
    episode_counter = 0

    @staticmethod
    def add_tf_summary(summary, episode=None):
        if episode is None:
            logger.writer.add_summary(summary, logger.episode_counter)
            logger.episode_counter += 1
        else:
            logger.writer.add_summary(summary, episode)
            logger.episode_counter = episode

    @staticmethod
    def add_tf_summary_with_last_episode(summary):
        logger.writer.add_summary(summary, logger.episode_counter)


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
