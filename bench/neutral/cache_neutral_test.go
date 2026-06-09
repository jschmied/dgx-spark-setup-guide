package cache

import (
	"sync"
	"testing"
	"time"
)

// Rule 5: Close must be safe to call multiple times (no panic).
func TestNeutralCloseIdempotent(t *testing.T) {
	c := New[string, int](10, time.Minute)
	c.Close()
	c.Close()
	c.Close()
}

// Rule 3: Get on an expired key returns false.
func TestNeutralExpiredMiss(t *testing.T) {
	c := New[string, int](10, time.Minute)
	defer c.Close()
	c.SetWithTTL("k", 1, 30*time.Millisecond)
	time.Sleep(80 * time.Millisecond)
	if v, ok := c.Get("k"); ok {
		t.Fatalf("expired key returned (%v,%v), want miss", v, ok)
	}
}

// Rule 1: at capacity, the least-recently-used live entry is evicted.
func TestNeutralLRUOrder(t *testing.T) {
	c := New[string, int](2, time.Minute)
	defer c.Close()
	c.Set("a", 1)
	c.Set("b", 2)
	if _, ok := c.Get("a"); !ok {
		t.Fatal("a should be live")
	} // a now MRU
	c.Set("c", 3) // capacity 2 -> evict LRU which is b
	if _, ok := c.Get("b"); ok {
		t.Fatal("b should have been evicted (LRU)")
	}
	if _, ok := c.Get("a"); !ok {
		t.Fatal("a should still be present")
	}
	if _, ok := c.Get("c"); !ok {
		t.Fatal("c should be present")
	}
}

// Rule 6: concurrency-safe (run under -race).
func TestNeutralRace(t *testing.T) {
	c := New[int, int](128, 50*time.Millisecond)
	defer c.Close()
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(base int) {
			defer wg.Done()
			for i := 0; i < 2000; i++ {
				k := (base*2000 + i) % 256
				c.Set(k, i)
				c.Get(k)
			}
		}(g)
	}
	wg.Wait()
}
