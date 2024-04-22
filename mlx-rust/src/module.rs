use half::f16;
use std::borrow::Cow;
use std::collections::HashMap;

use crate::io::SafeTensors;
use crate::stream::get_default_stream;
use crate::MLXArray;

pub trait Module {
    type Input;
    fn forward(&self, value: Self::Input) -> MLXArray;

    fn update_named_params(&mut self, prefix: &str, params: &mut HashMap<String, MLXArray>);

    fn update_by_safetensors<P: AsRef<std::path::Path>>(&mut self, filenames: &[P]) {
        let mut st_tensors: HashMap<String, MLXArray> = HashMap::new();
        for filename in filenames {
            let st = SafeTensors::new(filename.as_ref().to_str().unwrap(), get_default_stream());
            for (name, view) in st.data() {
                st_tensors.insert(name, view);
            }
        }
        // println!("embedding weight: {}", st_tensors.get("model.embed_tokens.weight"));
        self.update_named_params("", &mut st_tensors);
    }
}

pub trait WithParams {
    // fn gather_by_id(&self, params: &mut HashMap<usize, Tensor>);
    // fn update_by_id(&self, params: &mut HashMap<usize, Tensor>);

    fn gather_by_name(&self, params: &mut HashMap<String, MLXArray>, prefix: &str, name: &str);
    fn update_by_name(&mut self, params: &mut HashMap<String, MLXArray>, prefix: &str, name: &str);
}

impl WithParams for MLXArray {
    // fn gather_by_id(&self, params: &mut HashMap<usize, Tensor>) {
    //     params.insert(self.id(), self.clone());
    // }

    // fn update_by_id(&self, params: &mut HashMap<usize, Tensor>) {
    //     if let Some(t) = params.remove(&self.id()) {
    //         // todo: check if can promote type
    //         let t = t.to_dtype(self).to_device(self);
    //         self.replace_data(t);
    //     }
    // }

    fn gather_by_name(&self, params: &mut HashMap<String, MLXArray>, prefix: &str, name: &str) {
        let name = if prefix.is_empty() {
            name.into()
        } else {
            format!("{}.{}", prefix, name)
        };
        params.insert(name, self.clone());
    }

    fn update_by_name(&mut self, params: &mut HashMap<String, MLXArray>, prefix: &str, name: &str) {
        let name: Cow<'_, str> = if prefix.is_empty() {
            name.into()
        } else {
            format!("{}.{}", prefix, name).into()
        };
        if let Some(t) = params.remove(name.as_ref()) {
            // todo: check if can promote type
            // let t = t.to_dtype(self).to_device(self);
            // UnsafeCell
            //todo check shape
            *self = t;
        } else {
            panic!("parameter {} not found in params {:?}", name, params.keys());
        }
    }
}

impl<T> WithParams for Option<T>
where
    T: WithParams,
{
    // fn gather_by_id(&self, params: &mut HashMap<usize, Tensor>) {
    //     if let Some(t) = self {
    //         t.gather_by_id(params);
    //     }
    // }

    // fn update_by_id(&self, params: &mut HashMap<usize, Tensor>) {
    //     if let Some(t) = self {
    //         t.update_by_id(params);
    //     }
    // }

    fn gather_by_name(&self, params: &mut HashMap<String, MLXArray>, prefix: &str, name: &str) {
        if let Some(t) = self {
            t.gather_by_name(params, prefix, name);
        }
    }

    fn update_by_name(&mut self, params: &mut HashMap<String, MLXArray>, prefix: &str, name: &str) {
        if let Some(t) = self {
            t.update_by_name(params, prefix, name);
        }
    }
}

impl<T> WithParams for Vec<T>
where
    T: WithParams,
{
    // fn gather_by_id(&self, params: &mut HashMap<usize, Tensor>) {
    //     for t in self {
    //         t.gather_by_id(params);
    //     }
    // }

    // fn update_by_id(&self, params: &mut HashMap<usize, Tensor>) {
    //     for t in self {
    //         t.update_by_id(params);
    //     }
    // }

    fn gather_by_name(&self, params: &mut HashMap<String, MLXArray>, prefix: &str, name: &str) {
        for (i, t) in self.iter().enumerate() {
            let name = &format!("{}.{}", name, i);
            t.gather_by_name(params, prefix, name);
        }
    }

    fn update_by_name(&mut self, params: &mut HashMap<String, MLXArray>, prefix: &str, name: &str) {
        for i in 0..self.len() {
            let name = &format!("{}.{}", name, i);
            let t = &mut self[i];
            t.update_by_name(params, prefix, name);
        }
    }
}

impl<T> WithParams for T
where
    T: Module,
{
    // fn gather_by_id(&self, params: &mut HashMap<usize, Tensor>) {
    //     self.gather_params(params);
    // }

    // fn update_by_id(&self, params: &mut HashMap<usize, Tensor>) {
    //     self.update_params(params);
    // }

    fn gather_by_name(&self, params: &mut HashMap<String, MLXArray>, prefix: &str, name: &str) {
        // let p: Cow<'_, str> = if prefix.is_empty() {
        //     name.into()
        // } else {
        //     format!("{}.{}", prefix, name).into()
        // };
        // self.gather_named_params(&p, params)
        todo!()
    }

    fn update_by_name(&mut self, params: &mut HashMap<String, MLXArray>, prefix: &str, name: &str) {
        let p: Cow<'_, str> = if prefix.is_empty() {
            name.into()
        } else {
            format!("{}.{}", prefix, name).into()
        };
        self.update_named_params(&p, params)
    }
}
