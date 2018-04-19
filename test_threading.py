import threading
import numpy as np
import random
import time


'''
This is a complex example of how to use threads for my particular use case.

The idea is that we have many, many files that should be exhaustively read in
and batched, but not enough memory to hold them all.  So we read in some number
of files at once into a memory pool (that number equal to the number of threads),
and randomly shuffle between parts of the files in that pool to create a batch.

Numpy commands are written in C and do not block; threads allow for shared memory
which is important in this case, and though they don't allow for multiple processes
to be spun out, we are expecting that this code only consists of mainly of three pieces:
file i/o (# processes won't matter), numpy calls to edit our data (already non-blocking),
and writing results to shared memory (threads are best for this).  Because of this,
this structure should be the most optimal, as it should allow for threads to load in
files in the background as other threads that already have data in their memory continue
to yield.  This will especially be true if file reads/manipulations are not all the same
length and/or the number of threads is >> the batch size, so that we don't exhaust all
of the files at exactly the same time.

We assume we have files we're trying to read in, and a generator (generator_of_files)
that yields them in the order that matters.  These files are passed to a group
of threads that will each load one file off that list, process it in some way, and
take turns grabbing the numpy array that will ultimately be yielded to write one
part of it.

When a thread finishes with a file it will return to the generator and pop a new file
to read.  Each time a thread finishes updating the result numpy array (adding one example
to the batch and incrementing the batch counter 'curr_batch'), it notifies one other writer thread.
This 'one other thread gets notified' round robin approach is moderated by w_turn.
When the writer threads have filled a numpy array, they set read_ok and clear write_ok,
so that the consumer is notified and can yield.  When the consumer is done, it resets
write_ok so that the writers can resume filling up the shared memory results.

On startup, as well as when all the threads are stuck on file i/o (not unlikely),
the 'notify' for w_turn can fall on deaf ears (aka no threads cued and waiting to
write).  This means we need yet another shared event, so that if no write threads are
engaged, a new write thread will start writing even when none are notified.  This
is the job of the 'writing' event.

'''



files = ['data/a.txt','data/b.txt','data/c.txt','data/d.txt','data/e.txt','data/f.txt','data/g.txt','data/h.txt']

num_threads = 3
batch_size = 10
x = 5
y = 3


results = np.empty((batch_size,y,x), np.dtype('U'))
curr_batch = 0

gen_lock = threading.Lock()
w_turn = threading.Condition()
writing = threading.Event()
read_ok = threading.Event()
write_ok = threading.Event()

def generator_of_files():
    while(1):
        random.shuffle(files)
        for f in files:
            yield f

def threaded_writer(instance_gen_of_files, x, y, batch_size, i):
    #a = np.random.randint(0,10)
    global results
    global curr_batch

    while(1):
        #grab a file from the generator, use lock
        with gen_lock:
            f_str = next(instance_gen_of_files)

        #open it and formulate the value properly
        with open(f_str, 'r') as f:
            for line in f:
                val = np.array([[line]*x]*y)

                #take out the turn lock to write to batch, set rw_flags if necessary, update batch num, notify other threads
                with w_turn:
                    #wait for the thread to be tapped to write, wait for writing to be ok

                    if writing.is_set():
                        w_turn.wait()

                    writing.set()
                    write_ok.wait()
                    #update result, increase pointer into result
                    results[curr_batch,:,:] = val
                    curr_batch = curr_batch + 1
                    #if filled a batch, turn to read mode
                    if curr_batch == batch_size:
                        write_ok.clear()
                        read_ok.set()

                    #queue up next thread_writer
                    writing.clear()
                    w_turn.notify(n=1)

def consumer():
    '''meant to be initiated exactly once'''

    global results
    global curr_batch

    #wake up one write thread if this is the first time called
    #so they can start randomly filling a batch
    write_ok.set()

    while (1):
        #keep waiting on the read_ok to be set
        read_ok.wait()
        #when set, grab a copy of results, reset batch placeholder
        to_return = np.array(results, copy=True)
        curr_batch = 0
        #unset read_ok, set write_ok
        write_ok.set()
        read_ok.clear()
        #yield
        yield to_return



if __name__ == '__main__':

    test_gen = generator_of_files()

    for i in range(num_threads):
        worker = threading.Thread(target=threaded_writer, args=(test_gen, x, y, batch_size, i))
        worker.setDaemon(True)
        worker.start()

    consume_gen = consumer()

    start = time.time()

    num_yields = 1000
    for i in range(num_yields):
        next(consume_gen)

    end = time.time()

    print str(end-start) + ' seconds elapsed'
    print str((end-start)/float(num_yields)) + ' s avg yield'

