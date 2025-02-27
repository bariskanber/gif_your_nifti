"""Main entry point."""

import argparse
import gif_your_nifti.config as cfg
from gif_your_nifti import core, __version__
import warnings  # mainly for ignoring imageio warnings
warnings.filterwarnings("ignore")


def main():
    """Commandline interface."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'filename',  metavar='path', nargs='+',
        help="Path to image. Multiple paths can be provided."
        )
    parser.add_argument(
        '--mode', type=str, required=False,
        metavar=cfg.mode, default=cfg.mode,
        help="Gif creation mode. Available options are: 'normal', \
        'pseudocolor', 'depth', 'rgb'"
        )
    parser.add_argument(
        '--fps', type=int, required=False,
        metavar=cfg.fps, default=cfg.fps,
        help="Frames per second."
        )
    parser.add_argument(
        '--size', type=float, required=False,
        metavar=cfg.size, default=cfg.size,
        help="Image resizing factor."
        )
    parser.add_argument(
        '--cmap', type=str, required=False,
        metavar=cfg.cmap, default=cfg.cmap,
        help="Color map. Used only in combination with 'pseudocolor' mode."
        )
    parser.add_argument(
        '--frameskip', type=int, required=False,
        metavar=cfg.frameskip, default=cfg.frameskip,
        help="Will skip frames if >1 (useful for reducing GIF file size)."
        )
    parser.add_argument(
        '--colorcompressratio', type=int, required=False,
        metavar=cfg.colorcompressratio, default=cfg.colorcompressratio,
        help="Will compress colors if >1 (useful for reducing GIF file size)."
        )
    parser.add_argument(
        '--histeq', type=int, required=False,
        metavar=cfg.histeq, default=cfg.histeq,
        help="Will perform histogram equalization if set to 1."
        )

    args = parser.parse_args()
    cfg.mode = (args.mode).lower()
    cfg.size = args.size
    cfg.fps = args.fps
    cfg.cmap = args.cmap
    cfg.frameskip = args.frameskip
    cfg.colorcompressratio = args.colorcompressratio
    cfg.histeq = args.histeq

    # Welcome message
    welcome_str = '{} {}'.format('gif_your_nifti', __version__)
    welcome_decor = '=' * len(welcome_str)
    print('{}\n{}\n{}'.format(welcome_decor, welcome_str, welcome_decor))

    print('Selections:')
    print('  mode = {}'.format(cfg.mode))
    print('  size = {}'.format(cfg.size))
    print('  fps  = {}'.format(cfg.fps))
    print('  frameskip  = {}'.format(cfg.frameskip))
    print('  colorcompressratio  = {}'.format(cfg.colorcompressratio))
    print('  histeq = {}'.format(cfg.histeq))

    # Determine gif creation mode
    if cfg.mode in ['normal', 'pseudocolor', 'depth']:
        for f in args.filename:
            if cfg.mode == 'normal':
                core.write_gif_normal(f, cfg.size, cfg.fps, cfg.frameskip, cfg.colorcompressratio, cfg.histeq)
            elif cfg.mode == 'pseudocolor':
                print('  cmap = {}'.format(cfg.cmap))
                core.write_gif_pseudocolor(f, cfg.size, cfg.fps, cfg.cmap, cfg.frameskip)
            elif cfg.mode == 'depth':
                core.write_gif_depth(f, cfg.size, cfg.fps, cfg.frameskip)

    elif cfg.mode == 'rgb':
        if len(args.filename) != 3:
            raise ValueError('RGB mode requires 3 input files.')
        else:
            core.write_gif_rgb(args.filename[0], args.filename[1],
                               args.filename[2], cfg.size, cfg.fps, cfg.frameskip)
    else:
        raise ValueError("Unrecognized mode.")

    print('Finished.')


if __name__ == "__main__":
    main()
