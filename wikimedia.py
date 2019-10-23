import argparse
import copy
import datetime
import glob
import logging
import os
import re
import subprocess
from itertools import islice
from multiprocessing import Process, Queue, Value, cpu_count
from timeit import default_timer

from tqdm import tqdm

from asr.data.vocabulary import Vocabulary
from asr.lm.transform import PTBRTokenizer, Tokenizer
from asr.utils.io_utils import download_from_url

LOG = logging.getLogger('asr')


def download(lang='pt',
             data_dir=os.path.join('data', 'wikimedia',
                                   str(datetime.date.today()).replace('-', ''))):
    os.makedirs(data_dir, exist_ok=True)
    bz2_filepath = os.path.join(data_dir, '{}wiki-latest-pages-articles.xml.bz2'.format(lang))

    LOG.info('Saving data in "{}"'.format(data_dir))
    LOG.info('Chosen language: "{}"'.format(lang))

    if os.path.exists(bz2_filepath):
        print('File {} already downloaded. Skipping.'.format())
        return

    LOG.info("Continue to download (WARNING: This might be big and can take a long time!)")
    download_from_url(
        "https://dumps.wikimedia.org/{0}wiki/latest/{0}wiki-latest-pages-articles.xml.bz2".format(
            lang), bz2_filepath)


def extractor(lang='pt',
              data_dir=os.path.join('data', 'wikimedia',
                                    str(datetime.date.today()).replace('-', ''))):

    bz2_filepath = os.path.join(data_dir, '{}wiki-latest-pages-articles.xml.bz2'.format(lang))

    outdir = os.path.join(data_dir, '{}wiki-latest-pages-articles'.format(lang))
    outdir_wikiextractor = outdir + '-wikiextractor'

    if not os.path.isdir(outdir_wikiextractor):
        LOG.info('Calling wikiextractor.py. This may take a while...')
        subprocess.check_call([
            'python',
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'asr', 'lm', 'wikiextractor.py'),
            bz2_filepath, '-o', outdir_wikiextractor, '-q'
        ])
    else:
        LOG.info(f'Dir {outdir_wikiextractor} found. Skipping wikiextractor.py')


def preprocess(lang='pt',
               data_dir=os.path.join('data', 'wikimedia',
                                     str(datetime.date.today()).replace('-', '')),
               min_words=5,
               transliterate=False,
               keep_numbers=False,
               vocab=None,
               process_count=max(1,
                                 cpu_count() - 2)):
    outdir = os.path.join(data_dir, '{}wiki-latest-pages-articles'.format(lang))
    outdir_wikiextractor = outdir + '-wikiextractor'
    outdir_preprocessed = outdir + '-preprocessed'

    vocab = Vocabulary.from_file(vocab) if vocab else None

    if lang != 'pt':
        raise ValueError(f'Lang {lang} is not supported.')

    tok = Tokenizer(
        PTBRTokenizer(
            min_tokens=min_words,
            transliterate=transliterate,
            keep_numbers=keep_numbers,
            vocab=vocab))

    if not os.path.isdir(outdir_wikiextractor):
        LOG.error(f'Dir {outdir_wikiextractor} not found. Run first python wikimedia.py extractor')

    files = glob.glob(os.path.join(outdir_wikiextractor, '**', 'wiki_*'), recursive=True)

    process_count = max(1, process_count)
    maxsize = 10 * process_count

    worker_count = process_count

    # initialize jobs queue
    jobs_queue = Queue(maxsize=maxsize)
    pbar_q = Queue(maxsize=maxsize)

    # start progress bar process
    pbar_process = Process(target=_pbar_process, args=(pbar_q, len(files)))
    pbar_process.daemon = True
    pbar_process.start()

    # start worker processes
    LOG.info(f"Using {worker_count} extract processes.")
    extract_start = default_timer()

    workers = []
    for i in range(worker_count):
        extractor = Process(
            target=_extract_process,
            args=(jobs_queue, pbar_q, outdir_wikiextractor, outdir_preprocessed,
                  copy.deepcopy(tok)))
        extractor.daemon = True  # only live while parent process lives
        extractor.start()
        workers.append(extractor)

    # Mapper process
    page_num = 0
    for filepath in files:
        jobs_queue.put(filepath)  # goes to any available extract_process
        page_num += 1

    # signal termination
    for _ in workers:
        jobs_queue.put(None)

    # wait for workers to terminate
    for w in workers:
        w.join()

    pbar_q.put(None)
    pbar_process.join()

    extract_duration = default_timer() - extract_start
    extract_rate = page_num / extract_duration
    LOG.info(f"Finished {process_count}-process extraction of {page_num} "
             f"files in {extract_duration:.1f}s ({extract_rate:.1f}files/s)")


def _pbar_process(pbar_q, total, desc='Preprocessing files', position=0):
    with tqdm(total=total, desc=desc, position=position) as pbar:
        for n in iter(pbar_q.get, None):
            pbar.update(n)


def _extract_process(jobs_queue, pbar_q, outdir_wikiextractor, outdir_preprocessed, tok):

    for job in iter(jobs_queue.get, None):
        split_docs = r'\<doc(?:.*?)>([\s\S]+?)\<\/doc\>'

        filepath = job
        try:
            with open(filepath, encoding='utf8') as f:
                text = f.read()

            matches = re.finditer(split_docs, text)

            tokens = []
            for match_num, match in enumerate(matches):

                m = match.group(1)
                tokens.extend(tok.process_text(m))

            outfile = os.path.join(outdir_preprocessed,
                                   os.path.relpath(filepath, outdir_wikiextractor))
            os.makedirs(os.path.dirname(outfile), exist_ok=True)

            with open(outfile, 'w', encoding='utf8') as f:
                f.write('\n'.join([' '.join(t) for t in tokens]))
        except Exception as e:
            LOG.exception('Reading file {}'.format(filepath))
            raise e

        pbar_q.put(1)


def ascii_map():
    """https://bit.ly/2EpsMa8"""
    data = {}
    for num in range(256):
        h = num
        filename = f'x{num:03x}'
        try:
            mod = __import__('unidecode.' + filename, fromlist=True)
        except ImportError:
            pass
        else:
            for l, val in enumerate(mod.data):
                i = h << 8
                i += l
                if i >= 0x80:
                    data[i] = val
    return data


def _read_async(f_q, l_q, pbar_q, transliterate, vocab):

    vocab = Vocabulary.from_file(vocab) if vocab else None

    ascii_table = ascii_map()
    for filepath in iter(f_q.get, None):
        with open(filepath, encoding='utf8') as f:
            text = f.read()

        if transliterate:
            text = text.translate(ascii_table)

        lines = text.split('\n')

        if vocab:
            lines = filter(lambda line: all(c in vocab for c in line), lines)

        l_q.put([(l + '\n').encode('utf8') for l in lines])  # is it faster writing as binary?

        pbar_q.put(1)


def _write_async(f, q, num_files, num_lines):
    with tqdm(total=num_files, desc='Writing', position=1) as pbar:
        with open(f, 'wb') as out_f:
            for lines in iter(q.get, None):
                out_f.writelines(lines)
                num_lines.value += len(lines)
                pbar.update(1)


def split(lang='pt',
          data_dir=os.path.join('data', 'wikimedia',
                                str(datetime.date.today()).replace('-', '')),
          transliterate=False,
          vocab=None,
          suffix=''):

    outdir = os.path.join(data_dir, '{}wiki-latest-pages-articles'.format(lang))
    outdir_preprocessed = outdir + '-preprocessed'

    if not os.path.exists(outdir_preprocessed):
        raise ValueError('Path {} not found. Please run python wikimedia.py preprocess first'.
                         format(outdir_preprocessed))

    files = glob.glob(os.path.join(outdir_preprocessed, '**', 'wiki_*'), recursive=True)

    all_text_filepath = outdir + f'-text{suffix}.txt'

    LOG.info('Combining {} files'.format(len(files)))

    num_lines = Value('i', 0)
    process_count = max(1, max(1, cpu_count() - 1))
    worker_count = process_count

    # initialize jobs queue
    write_q = Queue(maxsize=len(files))
    read_q = Queue(maxsize=len(files) + worker_count)
    pbar_q = Queue(maxsize=len(files))

    # start progress bar process
    pbar_process = Process(target=_pbar_process, args=(pbar_q, len(files)))
    pbar_process.daemon = True
    pbar_process.start()

    # start worker processes
    LOG.info(f"Using {worker_count} read processes.")
    extract_start = default_timer()

    writer = Process(target=_write_async, args=(all_text_filepath, write_q, len(files), num_lines))
    writer.daemon = True
    writer.start()

    workers = []
    for i in range(worker_count):
        reader = Process(target=_read_async, args=(read_q, write_q, pbar_q, transliterate, vocab))
        reader.daemon = True  # only live while parent process lives
        reader.start()
        workers.append(reader)

    # Adding files in the Queue
    for f in files:
        read_q.put(f)

    # signal termination
    for _ in workers:
        read_q.put(None)

    for w in workers:
        w.join()

    write_q.put(None)
    writer.join()

    pbar_q.put(None)
    pbar_process.join()

    print()  # Fix tqdm
    num_lines = num_lines.value
    LOG.info('{} lines found'.format(num_lines))
    extract_duration = default_timer() - extract_start
    extract_rate = len(files) / extract_duration
    LOG.info(f"Finished {process_count}-process extraction of {len(files)} "
             f"files in {extract_duration:.1f}s ({extract_rate:.1f} files/s)")

    train_filepath = outdir + f'-train{suffix}.txt'
    test_filepath = outdir + f'-test{suffix}.txt'

    with open(all_text_filepath, 'rb') as f:
        for out_filepath, lines in zip([train_filepath, test_filepath],
                                       [islice(f, int(0.9 * num_lines)), f]):
            LOG.info('Saving {}'.format(out_filepath))
            with open(out_filepath, 'wb') as out_f:
                out_f.writelines(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        '--data-dir',
        default=os.path.join('data', 'wikimedia',
                             str(datetime.date.today()).replace('-', '')),
        type=str)
    common_parser.add_argument('--lang', default='pt', choices=['pt', 'en'], type=str)

    down_parser = subparsers.add_parser('download', parents=[common_parser])
    down_parser.set_defaults(func=download)

    ext_parser = subparsers.add_parser('extract', parents=[common_parser])
    ext_parser.set_defaults(func=extractor)

    prep_parser = subparsers.add_parser('preprocess', parents=[common_parser])
    prep_parser.add_argument('--min-words', type=int, default=5)
    prep_parser.add_argument('--vocab', default=None, type=str)
    prep_parser.add_argument(
        '--transliterate',
        action='store_true',
        default=False,
        help='Perform unicode transliteration to unicode')
    prep_parser.add_argument('--keep-numbers', action='store_true', default=False)
    prep_parser.add_argument('--process-count', default=max(1, cpu_count() - 2), type=int)
    prep_parser.set_defaults(func=preprocess)

    split_parser = subparsers.add_parser('split', parents=[common_parser])
    split_parser.add_argument('--suffix', default='', type=str)
    split_parser.add_argument('--vocab', default=None, type=str)
    split_parser.add_argument(
        '--transliterate',
        action='store_true',
        default=False,
        help='Perform unicode transliteration to unicode')
    split_parser.set_defaults(func=split)

    args = parser.parse_args()
    func = args.func

    del args.func
    func(**vars(args))
