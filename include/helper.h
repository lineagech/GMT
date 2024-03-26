#ifndef __HELPER_H__
#define __HELPER_H__

#define MEASURE_BW(func, qp) \
do {\
    uint32_t tail = (qp)->sq.in_ticket.load(simt::memory_order_acquire);\
    /*uint32_t head = (qp)->cq.out_ticket.load(simt::memory_order_acquire);*/\
    uint32_t head = (qp)->cq.head.load(simt::memory_order_acquire);\
    uint64_t start = clock64();\
    func;\
    uint64_t stop = clock64(); \
    uint32_t tail2 = (qp)->cq.head.load(simt::memory_order_acquire);\
    /*(qp)->cq.out_ticket.fetch_add(1, simt::memory_order_acq_rel);*/\
    printf("%llu:%u:%u\n", (stop-start), (tail-head), (tail2-head));\
} while(0)

#endif