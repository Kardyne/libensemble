.. _datastruct-worker-array:

worker array
=============

Stores information to inform the allocation function about the current state of
the workers. Workers can be in a variety of states. We take the following
convention:

=========================================   =======  ============  =======
Worker state                                 active  persis_state  blocked
=========================================   =======  ============  =======
idle worker                                    0          0           0   
active, nonpersistent sim                      1          0           0   
active, nonpersistent gen                      2          0           0   
active, persistent sim                         1          1           0   
active, persistent gen                         2          2           0   
waiting, persistent sim                        0          1           0   
waiting, persistent gen                        0          2           0   
worker blocked by some other calculation       1          0           1   
=========================================   =======  ============  =======

:Note:

* libE only receives from workers with 'active' nonzero
* libE only calls the alloc_f if some worker has 'active' zero

