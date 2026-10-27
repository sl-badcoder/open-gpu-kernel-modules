#!/usr/bin/env python3
"""Differentially test production prefetch traversal against its parent branch."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'kernel-open/nvidia-uvm/uvm_perf_prefetch.c'


def function(source, name):
    start = source.index(name + '(')
    start = source.rfind('\n', 0, start) + 1
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end] + '\n'


def macro(source, name):
    lines = source[source.index('#define ' + name + '('):].splitlines()
    result = []
    for line in lines:
        result.append(line)
        if not line.endswith('\\'):
            break
    return '\n'.join(result) + '\n'


PRELUDE = r'''
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
typedef uint16_t NvU16;
typedef uint16_t uvm_page_index_t;
typedef struct { uvm_page_index_t first, outer; } uvm_va_block_region_t;
typedef struct { unsigned long bitmap[512 / (8 * sizeof(unsigned long))]; } uvm_page_mask_t;
typedef struct { uvm_page_mask_t pages; uvm_page_index_t offset; NvU16 leaf_count; uint8_t level_count; } uvm_perf_prefetch_bitmap_tree_t;
typedef struct { int8_t level_idx; uvm_page_index_t node_idx; } uvm_perf_prefetch_bitmap_tree_iter_t;
#define UVM_ASSERT assert
#define UVM_ASSERT_MSG(c, ...) assert(c)
#define min(a,b) ((a) < (b) ? (a) : (b))
#define UVM_ALIGN_DOWN(v,a) ((v) & ~((a)-1))
static unsigned g_uvm_perf_prefetch_threshold;
static unsigned weights;
static unsigned long cases;
static bool bit(const unsigned long *b, unsigned p) { assert(p < 512); return (b[p/(8*sizeof(*b))] >> (p%(8*sizeof(*b)))) & 1; }
static void set(uvm_page_mask_t *m, unsigned p) { assert(p < 512); m->bitmap[p/(8*sizeof(unsigned long))] |= 1UL << (p%(8*sizeof(unsigned long))); }
static unsigned find_next_bit(const unsigned long *b, unsigned end, unsigned first) { assert(first <= end); while (first < end && !bit(b, first)) ++first; return first; }
static uvm_va_block_region_t uvm_va_block_region(unsigned a, unsigned b) { return (uvm_va_block_region_t){a,b}; }
static unsigned uvm_va_block_region_num_pages(uvm_va_block_region_t r) { return r.outer-r.first; }
static void uvm_page_mask_zero(uvm_page_mask_t *m) { memset(m,0,sizeof(*m)); }
static unsigned uvm_page_mask_region_weight(const uvm_page_mask_t *m, uvm_va_block_region_t r) { unsigned n=0; ++weights; for(unsigned i=r.first;i<r.outer;++i) n+=bit(m->bitmap,i); return n; }
static void uvm_page_mask_region_fill(uvm_page_mask_t *m, uvm_va_block_region_t r) { assert(r.first<=r.outer && r.outer<=512); for(unsigned i=r.first;i<r.outer;++i) set(m,i); }
'''

SCENARIOS = r'''
static void levels(uvm_perf_prefetch_bitmap_tree_t *t) { t->level_count=1; for(unsigned n=1;n<t->leaf_count;n*=2) ++t->level_count; }
static void compare(uvm_perf_prefetch_bitmap_tree_t *t, uvm_page_mask_t *f, uvm_va_block_region_t max, uvm_va_block_region_t faults) {
    uvm_page_mask_t a,b;
    uvm_perf_prefetch_bitmap_tree_t before=*t;
    weights=0; baseline(faults,max,t,f,&a); unsigned original=weights;
    assert(!memcmp(t,&before,sizeof(*t)));
    weights=0; compute_prefetch_mask(faults,max,t,f,&b);
    assert(weights<=original);
    assert(!memcmp(&a,&b,sizeof(a)) && !memcmp(t,&before,sizeof(*t)));
    ++cases;
}
static uint32_t seed=19;
static unsigned rng(void) { seed^=seed<<13; seed^=seed>>17; seed^=seed<<5; return seed; }
int main(void) {
    // All bitmap pairs and thresholds for small trees, including shifted geometry.
    for(unsigned len=1;len<=5;++len) for(unsigned offset=0;offset<=2 && offset<len;++offset)
    for(unsigned first=0;first<=3;first+=3) for(unsigned tree=0;tree<(1U<<(len+offset));++tree)
    for(unsigned faults=0;faults<(1U<<len);++faults) {
        uvm_perf_prefetch_bitmap_tree_t t={.offset=offset,.leaf_count=len+offset}; levels(&t);
        uvm_page_mask_t f={0};
        for(unsigned i=offset;i<len+offset;++i) if(tree & (1U<<i)) set(&t.pages,i);
        for(unsigned i=0;i<len;++i) if(faults & (1U<<i)) set(&f,first+i);
        for(g_uvm_perf_prefetch_threshold=0;g_uvm_perf_prefetch_threshold<=100;++g_uvm_perf_prefetch_threshold)
            compare(&t,&f,uvm_va_block_region(first,first+len),uvm_va_block_region(first,first+len));
    }
    // Large and truncated trees; independent fault masks and partial fault regions.
    for(unsigned trial=0;trial<20000;++trial) {
        unsigned len=1+rng()%512, first=rng()%(513-len), offset=rng()%(1+min(15,min(len-1,512-len)));
        uvm_perf_prefetch_bitmap_tree_t t={.offset=offset,.leaf_count=len+offset}; levels(&t);
        uvm_page_mask_t f={0};
        unsigned density=rng()%101;
        for(unsigned i=offset;i<len+offset;++i) if(rng()%100<density) set(&t.pages,i);
        for(unsigned i=0;i<len;++i) if(rng()%100<density) set(&f,first+i);
        unsigned begin=first+rng()%(len+1), end=begin+rng()%(first+len-begin+1);
        g_uvm_perf_prefetch_threshold=rng()%101;
        compare(&t,&f,uvm_va_block_region(first,first+len),uvm_va_block_region(begin,end));
    }
    uvm_perf_prefetch_bitmap_tree_t t={.leaf_count=512}; levels(&t);
    uvm_page_mask_t f={0}, a,b;
    for(unsigned i=0;i<256;++i) {set(&t.pages,i);set(&f,i);}
    g_uvm_perf_prefetch_threshold=51;
    weights=0;baseline(uvm_va_block_region(0,512),uvm_va_block_region(0,512),&t,&f,&a);unsigned original=weights;
    weights=0;compute_prefetch_mask(uvm_va_block_region(0,512),uvm_va_block_region(0,512),&t,&f,&b);
    assert(!memcmp(&a,&b,sizeof(a)) && original==2560 && weights==10);
    printf("%lu equivalent masks; dense-half bitmap weights: %u -> %u\n",cases,original,weights);
    // Fault region ends inside the chosen node: do not overrun equality iterator.
    compare(&t,&f,uvm_va_block_region(0,512),uvm_va_block_region(7,63));
    return 0;
}
'''


def main():
    source = (ROOT / SOURCE).read_text()
    original = subprocess.check_output(['git', 'show', '03740819:' + SOURCE], cwd=ROOT, text=True)
    utils = (ROOT / 'kernel-open/nvidia-uvm/uvm_perf_utils.h').read_text()
    header = (ROOT / 'kernel-open/nvidia-uvm/uvm_perf_prefetch.h').read_text()
    block = (ROOT / 'kernel-open/nvidia-uvm/uvm_va_block.h').read_text()
    code = PRELUDE
    for name in ('uvm_perf_tree_iter_max_leaves', 'uvm_perf_tree_iter_leaf_range_start', 'uvm_perf_tree_iter_leaf_range'):
        code += macro(utils, name)
    code += macro(header, 'uvm_perf_prefetch_bitmap_tree_traverse_counters')
    for name in ('uvm_va_block_first_page_in_mask', 'uvm_va_block_next_page_in_mask'):
        code += function(block, name)
    code += macro(block, 'for_each_va_block_page_in_region_mask')
    for name in ('uvm_perf_prefetch_bitmap_tree_iter_init', 'uvm_perf_prefetch_bitmap_tree_iter_get_range', 'uvm_perf_prefetch_bitmap_tree_iter_get_count', 'compute_prefetch_region', 'compute_prefetch_mask'):
        code += function(source, name)
    code += function(original, 'compute_prefetch_mask').replace('compute_prefetch_mask(', 'baseline(')
    code += SCENARIOS
    with tempfile.TemporaryDirectory(prefix='prefetch-region-skip-') as directory:
        c = Path(directory) / 'test.c'
        binary = Path(directory) / 'test'
        c.write_text(code)
        subprocess.run(['cc', '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined', '-g', str(c), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
