from datetime import datetime
from math import ceil
import itertools
import yaml
import time
import ray
import sys
import gc
import os

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# "Background" code for the user

# This class will be the key to able the user to deserialize the data transparently
class RayList(list):
    def __call__(self, index): # Square brackets operation to obtain the data behind the references.
        item = super().__getitem__(index)
        if isinstance(index, slice):
            free(item)
        else:
            free([item])

    def __getitem__(self, index): # Square brackets operation to obtain the data behind the references.
        item = super().__getitem__(index)
        if isinstance(index, slice):
            return ray.get(RayList(item))
        else:
            return ray.get(item)

class Reisa:
    def __init__(self, file, address):
        self.iterations = 0
        self.mpi_per_node = 0
        self.mpi = 0
        self.datasize = 0
        self.workers = 0
        self.actor = None
        
        # Init Ray
        if os.environ.get("REISA_DIR"):
            ray.init("ray://"+address+":10001", runtime_env={"working_dir": os.environ.get("REISA_DIR")})
        else:
            ray.init("ray://"+address+":10001", runtime_env={"working_dir": os.environ.get("PWD")})
       
        # Get the configuration of the simulatin
        with open(file, "r") as stream:
            try:
                data = yaml.safe_load(stream)
                self.iterations = data["MaxtimeSteps"]
                self.mpi_per_node = data["mpi_per_node"]
                self.mpi = data["parallelism"]["height"] * data["parallelism"]["width"]
                self.workers = data["workers"]
                self.datasize = data["global_size"]["height"] * data["global_size"]["width"]
            except yaml.YAMLError as e:
                eprint(e)

        return
    
    def get_result(self, process_func, iter_func, global_func=None, selected_iters=None, kept_iters=None, timeline=False):
            
        max_tasks = ray.available_resources()['compute']
        results = list()
        actor = self.get_actor()
        
        if selected_iters is None:
            selected_iters = [i for i in range(self.iterations)]
        if kept_iters is None:
            kept_iters = self.iterations

        # process_task = ray.remote(max_retries=-1, resources={"compute":1}, scheduling_strategy="DEFAULT")(process_func)
        # iter_task = ray.remote(max_retries=-1, resources={"compute":1, "transit":0.5}, scheduling_strategy="DEFAULT")(iter_func)

        @ray.remote(max_retries=-1, resources={"compute":1}, scheduling_strategy="DEFAULT")
        def process_task(rank: int, i: int, queue):
            return process_func(rank, i, queue)
            
        iter_ratio=1/(ceil(max_tasks/self.mpi)*2)

        @ray.remote(max_retries=-1, resources={"compute":1, "transit":iter_ratio}, scheduling_strategy="DEFAULT")
        def iter_task(i: int, actor):
            current_result = actor.trigger.remote(process_task, i)
            current_result = ray.get(current_result)
            
            if i >= kept_iters-1:
                actor.free_mem.remote(current_result, i-kept_iters+1)
            
            return iter_func(i, RayList(current_result))

        start = time.time() # Measure time
        results = [iter_task.remote(i, actor) for i in selected_iters]
        ray.wait(results, num_returns=len(results)) # Wait for the results
        eprint("{:<21}".format("EST_ANALYTICS_TIME:") + "{:.5f}".format(time.time()-start) + " (avg:"+"{:.5f}".format((time.time()-start)/self.iterations)+")")
        if global_func:
            return global_func(RayList(results))
        else:
            tmp = ray.get(results)
            output = {} # Output dictionary

            for i, _ in enumerate(selected_iters):
                if tmp[i] is not None:
                    output[selected_iters[i]] = tmp[i]
            
            if timeline:
                ray.timeline(filename="timeline-client.json")

            return output

    # Get the actor created by the simulation
    def get_actor(self):
        timeout = 60
        start_time = time.time()
        error = True
        self.actor = None
        while error:
            try:
                self.actor = ray.get_actor("global_actor", namespace="mpi")
                error = False
            except Exception as e:
                self.actor = None
                end_time = time.time()
                elapsed_time = end_time - start_time
                if elapsed_time >= timeout:
                    raise Exception("Cannot get the Ray actor. Client is exiting")
            time.sleep(1)

        return self.actor

    # Erase iterations from simulation memory
    def shutdown(self):
        if self.actor:
            ray.kill(self.actor)
            ray.shutdown()