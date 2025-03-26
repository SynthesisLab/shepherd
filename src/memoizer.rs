use std::collections::HashMap;
use std::hash::Hash;

pub struct Memoizer<K, V, F>
where
    K: Eq + Hash + Clone,
    V: Clone,
    F: Fn(&K) -> V,
{
    cache: HashMap<K, V>,
    compute: F,
}

impl<K, V, F> Memoizer<K, V, F>
where
    K: Eq + Hash + Clone,
    V: Clone,
    F: Fn(&K) -> V,
{
    pub fn new(compute: F) -> Self {
        Self {
            cache: HashMap::new(),
            compute,
        }
    }

    pub fn get(&mut self, key: K) -> V {
        if let Some(value) = self.cache.get(&key) {
            return value.clone();
        }
        let value = (self.compute)(&key);
        self.cache.insert(key.clone(), value.clone());
        value
    }
}
