const COLD_WORKING_SET_BYTES: usize = 256 << 20;

// On iPhone the jetsam limit of a headless devicectl-launched process is far
// below the default 256 MB working set; UZU_COLD_POOL_MB shrinks it there.
fn cold_working_set_bytes() -> usize {
    std::env::var("UZU_COLD_POOL_MB")
        .ok()
        .and_then(|mb| mb.parse::<usize>().ok())
        .map(|mb| mb << 20)
        .unwrap_or(COLD_WORKING_SET_BYTES)
}

pub struct ColdPool<T, F: FnMut() -> T> {
    bytes_per_copy: usize,
    alloc: F,
    copies: Vec<T>,
    next: usize,
}

impl<T, F: FnMut() -> T> ColdPool<T, F> {
    pub fn new(
        bytes_per_copy: usize,
        alloc: F,
    ) -> Self {
        Self {
            bytes_per_copy,
            alloc,
            copies: Vec::new(),
            next: 0,
        }
    }

    pub fn next_mut(&mut self) -> &mut T {
        if self.copies.is_empty() {
            let count = copy_count(cold_working_set_bytes(), self.bytes_per_copy);
            self.copies = (0..count).map(|_| (self.alloc)()).collect();
        }
        let index = self.next;
        self.next = (index + 1) % self.copies.len();
        &mut self.copies[index]
    }
}

pub(crate) fn copy_count(
    working_set: usize,
    bytes_per_copy: usize,
) -> usize {
    working_set.div_ceil(bytes_per_copy).max(1)
}
