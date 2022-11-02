#![deny(unsafe_op_in_unsafe_fn)]

//! An simple atomic reference counter.

use std::{
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicUsize, Ordering},
};

/// An atomic reference counter.
#[repr(transparent)]
#[derive(Debug, Default)]
pub struct AtomicBorrow {
    borrow: AtomicUsize,
}

impl AtomicBorrow {
    /// The mask for the shared borrow count.
    pub const SHARED_MASK: usize = usize::MAX >> 1;
    /// The mask for the unique borrow bit.
    pub const UNIQUE_MASK: usize = !Self::SHARED_MASK;

    const SPIN_COUNT: usize = 1 << 10;

    /// Creates a new `AtomicBorrow`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            borrow: AtomicUsize::new(0),
        }
    }

    /// Returns number of shared borrows.
    #[inline]
    pub fn shared_count(&self) -> usize {
        self.borrow.load(Ordering::Acquire) & Self::SHARED_MASK
    }

    /// Returns true if `self` is uniquely borrowed.
    #[inline]
    pub fn is_unique(&self) -> bool {
        self.borrow.load(Ordering::Acquire) & Self::UNIQUE_MASK == 0
    }

    /// Returns true if `self` is borrowed in any way.
    #[inline]
    pub fn is_borrowed(&self) -> bool {
        self.borrow.load(Ordering::Acquire) != 0
    }

    /// Tries to acquire a shared reference.
    ///
    /// Returns `true` if the reference was acquired.
    #[inline]
    pub fn borrow(&self) -> bool {
        let prev = self.borrow.fetch_add(1, Ordering::Acquire);

        if prev & Self::SHARED_MASK == Self::SHARED_MASK {
            panic!("borrow counter overflowed");
        }

        if prev & Self::UNIQUE_MASK != 0 {
            // we're already uniquely borrowed, so undo the increment and return false
            self.borrow.fetch_sub(1, Ordering::Release);
            false
        } else {
            true
        }
    }

    /// Tries to acquire a unique reference.
    ///
    /// Returns `true` if the reference was acquired.
    #[inline]
    pub fn borrow_mut(&self) -> bool {
        self.borrow
            .compare_exchange(0, Self::UNIQUE_MASK, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    /// Releases a shared reference.
    ///
    /// # Panics.
    /// * If `self` is not borrowed. Only with `debug_assertions` enabled.
    /// * If `self` is uniquely borrowed. Only with `debug_assertions` enabled.
    #[inline]
    pub fn release(&self) {
        let prev = self.borrow.fetch_sub(1, Ordering::Release);
        debug_assert_ne!(
            prev, 0,
            "borrow counter underflow, this means you released more times than you borrowed"
        );
        debug_assert_eq!(
            prev & Self::UNIQUE_MASK,
            0,
            "shared release of unique borrow"
        );
    }

    /// Releases a unique reference.
    ///
    /// # Panics.
    /// * If `self` is not uniquely borrowed. Only with `debug_assertions` enabled.
    #[inline]
    pub fn release_mut(&self) {
        let prev = self.borrow.fetch_and(!Self::UNIQUE_MASK, Ordering::Release);
        debug_assert_ne!(
            prev & Self::UNIQUE_MASK,
            0,
            "unique release of shared borrow"
        );
    }

    /// Spins until a shared reference can be acquired.
    #[inline]
    pub fn spin_borrow(&self) {
        for _ in 0..Self::SPIN_COUNT {
            if self.borrow() {
                return;
            }

            std::hint::spin_loop();
        }

        while !self.borrow() {
            std::thread::yield_now();
        }
    }

    /// Spins until a unique reference can be acquired.
    #[inline]
    pub fn spin_borrow_mut(&self) {
        for _ in 0..Self::SPIN_COUNT {
            if self.borrow_mut() {
                return;
            }

            std::hint::spin_loop();
        }

        while !self.borrow_mut() {
            std::thread::yield_now();
        }
    }
}

/// A guard that releases a shared reference when dropped.
pub struct SharedGuard<'a, T> {
    data: *const T,
    borrow: &'a AtomicBorrow,
}

impl<'a, T> SharedGuard<'a, T> {
    /// Creates a new [`SharedGuard`].
    #[inline]
    pub fn new(data: &'a T, borrow: &'a AtomicBorrow) -> Self {
        Self { data, borrow }
    }

    /// Tries to borrow the data.
    ///
    /// # Safety
    /// * Any borrows of `data` must be registered with `borrow`.
    /// * `data` must be a valid pointer for the entire lifetime of `self`.
    #[inline]
    pub unsafe fn try_new(data: *const T, borrow: &'a AtomicBorrow) -> Option<Self> {
        if borrow.borrow() {
            Some(Self { data, borrow })
        } else {
            None
        }
    }

    /// Spins until the data can be borrowed.
    ///
    /// # Safety
    /// * Any borrows of `data` must be registered with `borrow`.
    /// * `data` must be a valid pointer for the entire lifetime of `self`.
    #[inline]
    pub unsafe fn spin(data: *const T, borrow: &'a AtomicBorrow) -> Self {
        borrow.spin_borrow();
        Self { data, borrow }
    }

    /// Gets the inner [`AtomicBorrow`].
    #[inline]
    pub fn get_borrow(&self) -> &AtomicBorrow {
        self.borrow
    }

    /// Gets the inner data.
    #[inline]
    pub fn ptr(&self) -> *const T {
        self.data
    }

    /// Gets the inner data without releasing the borrow.
    #[inline]
    pub fn forget(self) -> *const T {
        let ptr = self.ptr();
        std::mem::forget(self);
        ptr
    }
}

impl<'a, T> Deref for SharedGuard<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.data }
    }
}

impl<'a, T> Drop for SharedGuard<'a, T> {
    #[inline]
    fn drop(&mut self) {
        self.borrow.release();
    }
}

/// A guard that releases a unique reference when dropped.
pub struct UniqueGuard<'a, T> {
    data: *mut T,
    borrow: &'a AtomicBorrow,
}

impl<'a, T> UniqueGuard<'a, T> {
    /// Creates a new [`UniqueGuard`].
    #[inline]
    pub fn new(data: &'a mut T, borrow: &'a AtomicBorrow) -> Self {
        Self { data, borrow }
    }

    /// Tries to borrow the data.
    ///
    /// # Safety
    /// * Any borrows of `data` must be registered with `borrow`.
    /// * `data` must be a valid pointer for the entire lifetime of `self`.
    #[inline]
    pub unsafe fn try_new(data: *mut T, borrow: &'a AtomicBorrow) -> Option<Self> {
        if borrow.borrow_mut() {
            Some(Self { data, borrow })
        } else {
            None
        }
    }

    /// Spins until the data can be borrowed.
    ///
    /// # Safety
    /// * Any borrows of `data` must be registered with `borrow`.
    /// * `data` must be a valid pointer for the entire lifetime of `self`.
    #[inline]
    pub unsafe fn spin(data: *mut T, borrow: &'a AtomicBorrow) -> Self {
        borrow.spin_borrow_mut();
        Self { data, borrow }
    }

    /// Gets the inner [`AtomicBorrow`].
    #[inline]
    pub fn get_borrow(&self) -> &AtomicBorrow {
        self.borrow
    }

    /// Gets the inner data.
    #[inline]
    pub fn ptr(&self) -> *mut T {
        self.data
    }

    /// Gets the inner data without releasing the borrow.
    #[inline]
    pub fn forget(self) -> *mut T {
        let ptr = self.ptr();
        std::mem::forget(self);
        ptr
    }
}

impl<'a, T> Deref for UniqueGuard<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.data }
    }
}

impl<'a, T> DerefMut for UniqueGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.data }
    }
}

impl<'a, T> Drop for UniqueGuard<'a, T> {
    #[inline]
    fn drop(&mut self) {
        self.borrow.release_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_borrow() {
        let borrow = AtomicBorrow::new();

        assert!(borrow.borrow());
        assert!(borrow.borrow());

        assert!(!borrow.borrow_mut());

        borrow.release();
        borrow.release();

        assert!(borrow.borrow_mut());

        assert!(!borrow.borrow());

        borrow.release_mut();
    }
}
