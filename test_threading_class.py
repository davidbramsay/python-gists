import threading
import numpy as np
import random
import time


'''
Take test_threading example, and turn it into a useful class.

It should take a file generator, preprocessing function, and num_threads, and
output a generator.  It hsould also pass in batch_size and output size from
the preprocessing function.

The preprocessing function should take the filename string and yield numpy arrays
of proper output dimensions.
'''


class ThreadManager(object):


    def __init__(self, file_generator, preprocess_generator, output_dims=(256,20), batch_size=32, num_threads=64):
        self.preprocess_generator = preprocess_generator
        self.output_dims = output_dims
        self.batch_size = batch_size
        self.num_threads = num_threads

        self._file_gen = file_generator
        self._results = np.empty((batch_size,)+output_dims, dtype=np.dtype('U'))
        self._curr_batch = 0

        #locks to manage the threads
        self._gen_lock = threading.Lock()
        self._w_turn = threading.Condition()
        self._writing = threading.Event()
        self._read_ok = threading.Event()
        self._write_ok = threading.Event()

        self._workers = []
        for i in range(num_threads):
            self._workers.append(threading.Thread(target=self._threaded_writer, args=()))
            self._workers[-1].setDaemon(True)
            self._workers[-1].start()


    def _threaded_writer(self):
        '''many of these will take turns writing to the shared results array, one example at
        a time, until batch_size examples have been written.  This should greatly diminish
        any bottlenecks due to file i/o or numpy preprocessing.'''


        while(1):
            #grab a file from the generator, use lock
            with self._gen_lock:
                f_str = next(self._file_gen)

            our_generator = self.preprocess_generator(f_str, self.output_dims)

            while(1):

                try:
                    val = next(our_generator)
                except StopIteration as e:
                    break

                with self._w_turn:

                    #if someone is writing, wait your turn (for w_turn to notify you)
                    if self._writing.is_set():
                        self._w_turn.wait()

                    #it's our turn!  Set the writing flag and make sure our shared memory is ready to be written
                    self._writing.set()
                    self._write_ok.wait()

                    #update result, increase pointer for current example in batch of result
                    self._results[self._curr_batch,...] = val
                    self._curr_batch = self._curr_batch + 1

                    #if we've filled a batch, put ourselves in read mode and turn off write mode
                    if self._curr_batch == self.batch_size:
                        self._write_ok.clear()
                        self._read_ok.set()

                    #we're done writing, notify the next thread_writer
                    self._writing.clear()
                    self._w_turn.notify(n=1)


    def gen(self):
        '''meant to be initiated exactly once, this will yield batches from the
        shared results array that is being written into by many threaded_writer
        processes.  It is a generator.'''

        #wake up one write thread if this is the first time called
        #so they can start randomly filling a batch
        self._write_ok.set()

        while (1):
            #wait to be told it's ok to read (aka we have a full results array)
            self._read_ok.wait()

            #our turn! grab a copy of results, reset batch placeholder
            to_return = np.array(self._results, copy=True)
            self._curr_batch = 0

            #unset read_ok, set write_ok so the writers can continue
            self._write_ok.set()
            self._read_ok.clear()

            #yield
            yield to_return



def generator_of_files(files):
    '''gives a file from a list of data files, in a meaningful order.  Here we are simply
    making sure we go through one full epoch before repeatings, and shuffling the order
    between epochs.'''

    while(1):
        random.shuffle(files)
        for f in files:
            yield f

def preprocess_generator(filename, output_dims):
    '''take in a filename, load it, preprocess it, and generate chunks of data
    '''

    output_array = np.empty(output_dims, np.dtype('U'))

    with open(filename, 'r') as f:
        for line in f:
            output_array.fill(line)

    for i in range(np.random.randint(8,10)):
        yield output_array


if __name__ == '__main__':

    files = ['data/a.txt','data/b.txt','data/c.txt','data/d.txt','data/e.txt','data/f.txt','data/g.txt','data/h.txt']

    #create a shared instance of the file generator
    shared_file_gen_instance = generator_of_files(files)

    test = ThreadManager(shared_file_gen_instance, preprocess_generator, (5,2,3), 5, 2)
    gen = test.gen()

    test2 = ThreadManager(shared_file_gen_instance, preprocess_generator, (5,2), 5, 2)
    gen2 = test2.gen()

    num_yields = 3
    for i in range(num_yields):
        print next(gen)
        print next(gen2)
        print '-'*20
        print '-'*20

