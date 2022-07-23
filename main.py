import cv2
import click 


import numpy as np
import operator as op 
import itertools as it, functools as ft 

import zmq 
import multiprocessing as mp 

from os import path, listdir 
from glob import glob 
from rich.progress import Progress, track 


from typing import List, Tuple, Dict, Callable
from time import sleep
from sys import stdout 
from loguru import logger 

log_format = [
	'<W><k>{time: YYYY-MM-DD hh:mm:ss}</k></W>',
	'<c>{file:^15}</c>',
	'<w>{function:^20}</w>',
	'<e>{line:03d}</e>',
	'<r>{level:^10}</r>',
	'<Y><k>{message:<50}</k></Y>'
]

log_separator = ' | '

logger.remove()
logger.add(
	sink=stdout, 
	level='TRACE', 
	format=log_separator.join(log_format)
)

"""
    python main.py sequential-pipeline --path2source  --path2dump  
    python main.py parallel-pipeline --path2source --path2dump --nb_workers --port 
"""


def image_processing(worker_id:int, path2images:List[str], path2dump:str, server_address:str, server_readyness:mp.Event, workers_counter:mp.Value):
    # initialization 
    header = f'worker {worker_id:03d}'
    nb_images = len(path2images)
    logger.debug(f'{header} is ready')
    logger.debug(f'{header} has got {nb_images:04d} images to process')
    server_readyness.wait()  # wait until server send the signal to start proessing 
    logger.debug(f'{header} got the signal from server')
    ZMQ_INIT = 0
    try:
        
        ctx = zmq.Context()
        dealer_socket = ctx.socket(zmq.DEALER)  
        dealer_socket.connect(server_address)

        ZMQ_INIT = 1  
        logger.success(f'{header} has finished to initialize zmq')

        for path2image in path2images:
            # read data
            bgr_image = cv2.imread(path2image, cv2.IMREAD_COLOR)

            # transform data 
            resized_image = cv2.resize(bgr_image, (256, 256))
            basepath, _ = path.split(path2image)
            path2resized_image = path2image.replace(basepath, path2dump)
            
            # serialize data 
            cv2.imwrite(path2resized_image, resized_image)

            # notify server
            dealer_socket.send_multipart([b'', f'{worker_id}'.encode()])
            _ = dealer_socket.recv_multipart()
        # end for loop over images 

    except KeyboardInterrupt as e:
        pass 

    except Exception as e:
        logger.error(e)
    finally:
        if ZMQ_INIT == 1:
            dealer_socket.close()
            ctx.term()

        with workers_counter.get_lock():  # notify server that the job was done ...!
            workers_counter.value = workers_counter.value - 1
            logger.success(f'{header} has released all zmq ressources')

@click.group(chain=False, invoke_without_command=True)
@click.pass_context
def group(ctx):
    subcommand = ctx.invoked_subcommand 
    if subcommand is not None:
        logger.debug(f'{subcommand} was called')


@group.command()
@click.option('--path2source', help='path to the directory where images are saved', type=click.Path(True))
@click.option('--path2dump', help='path to the directory where transformed images will be saved', type=click.Path(True))
def sequential_pipeline(path2source, path2dump):
    file_paths = sorted(glob(path.join(path2source, '*.jpg')))[:10000]
    nb_images = len(file_paths)
    logger.debug(f'{nb_images:05d} images were found at {path2source}')

    for path2image in track(file_paths, 'image processing'):
        bgr_image = cv2.imread(path2image, cv2.IMREAD_COLOR)
        resized_image = cv2.resize(bgr_image, (256, 256))
        basepath, _ = path.split(path2image)
        path2resized_image = path2image.replace(basepath, path2dump)
        cv2.imwrite(path2resized_image, resized_image)

    logger.success('all images were processed...!')

@group.command()
@click.option('--path2source', help='path to the directory where images are saved', type=click.Path(True))
@click.option('--path2dump', help='path to the directory where transformed images will be saved', type=click.Path(True))
@click.option('--nb_workers', help='number of workers to lunch', type=int, default=8)
@click.option('--port', help='port of the server', type=int, default=8500)
def parallel_pipeline(path2source, path2dump, nb_workers, port):
    file_paths = sorted(glob(path.join(path2source, '*.jpg')))[:10000]
    nb_images = len(file_paths)
    logger.debug(f'{nb_images:05d} images were found at {path2source}')
    packets = np.array_split(file_paths, nb_workers)

    server_readyness = mp.Event()
    workers_counter = mp.Value('i', nb_workers)

    tasks_acc = []
    processes_acc = []
    for packet_id, packet in enumerate(packets):
        prs = mp.Process(
            target=image_processing, 
            args=[packet_id, packet, path2dump, f'tcp://localhost:{port}', server_readyness, workers_counter]
        )
        processes_acc.append(prs)
        processes_acc[-1].start()  # lunch the process 
        tasks_acc.append((packet_id, len(packet)))
    # end for loop over packets 

    ZMQ_INIT = 0 
    try:
        ctx = zmq.Context()
        router_socket = ctx.socket(zmq.ROUTER)
        router_socket.bind(f'tcp://*:{port}')

        router_poller = zmq.Poller()
        router_poller.register(router_socket, zmq.POLLIN)
        server_readyness.set()  # notify worker that server is ready 
        logger.success('server initialization complete...!')

        with Progress() as progressor:
            for packet_id, nb_items in tasks_acc:
                progressor.add_task(f'worker {packet_id:03d} image processing', total=nb_items)

            keep_routing = True 
            while keep_routing:
                if workers_counter.value == 0:
                    keep_routing = False
            
                incoming_events = dict(router_poller.poll(10))
                router_socket_status = incoming_events.get(router_socket, None)
                if router_socket_status is not None:
                    if router_socket_status == zmq.POLLIN: 
                        caller, delimeter, encoded_msg = router_socket.recv_multipart()
                        worker_id = int(encoded_msg.decode())
                        progressor.update(worker_id, advance=1)
                        router_socket.send_multipart([caller, delimeter, b''])
            # end loop over routing 

    except KeyboardInterrupt as e:
        pass 
    except Exception as e:
        logger.error(e)
    finally:
        if ZMQ_INIT == 1:
            router_poller.unregister(router_socket)
            router_socket.close()
            ctx.term()
            logger.success('server has released zmq ressources')
            
    logger.debug('end of program')

if __name__ == '__main__':
    group(obj={})