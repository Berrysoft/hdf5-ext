use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro_error2::abort;
use quote::quote;
use syn::{
    AttrStyle, Attribute, Data, DeriveInput, Field, Fields, GenericParam, Generics, Ident,
    parse_str,
};

struct PreDerive {
    attrs: Vec<Attribute>,
    struct_name: Ident,
    generics: Generics,
    data: Data,
    generic_inputs: proc_macro2::TokenStream,
    dst_crate_name: proc_macro2::TokenStream,
}

fn pre_derive(input: TokenStream) -> PreDerive {
    let struct_input: DeriveInput = syn::parse(input).unwrap();
    let generics = struct_input.generics;
    let generic_inputs = generics
        .params
        .iter()
        .map(|p| match p {
            GenericParam::Type(p) => {
                let ident = &p.ident;
                quote!(#ident)
            }
            GenericParam::Lifetime(p) => {
                let life = &p.lifetime;
                quote!(#life)
            }
            GenericParam::Const(p) => {
                let ident = &p.ident;
                quote!(#ident)
            }
        })
        .collect::<Vec<_>>();
    let generic_inputs = if generic_inputs.is_empty() {
        quote!()
    } else {
        quote!(<#(#generic_inputs,)*>)
    };

    let dst_crate_name = match crate_name("hdf5-dst").unwrap() {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let name = parse_str::<Ident>(&name).unwrap();
            quote!(::#name)
        }
    };

    PreDerive {
        attrs: struct_input.attrs,
        struct_name: struct_input.ident,
        generics,
        data: struct_input.data,
        generic_inputs,
        dst_crate_name,
    }
}

fn find_repr(attrs: &[Attribute], expected: &[&str]) -> Option<Ident> {
    let mut repr = None;
    for attr in attrs.iter() {
        if attr.style != AttrStyle::Outer {
            continue;
        }
        if !attr.path().is_ident("repr") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if expected.iter().any(|s| meta.path.is_ident(s)) {
                if repr.is_some() {
                    abort!(meta.path, "ambiguous repr attribute");
                } else {
                    repr = meta.path.get_ident().cloned();
                }
            }
            Ok(())
        })
        .ok()?;
    }
    repr
}

#[proc_macro_derive(H5TypeUnsized)]
pub fn derive_h5type_unsized(input: TokenStream) -> TokenStream {
    let PreDerive {
        attrs,
        struct_name,
        generics,
        data,
        generic_inputs,
        dst_crate_name,
    } = pre_derive(input);

    find_repr(&attrs, &["C", "transparent"]).expect("expect repr(C) or repr(transparent)");

    let calculate_type = match data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => {
                let fields = fields.named.into_iter().collect::<Vec<_>>();
                let stats = map_compound(fields, &dst_crate_name);
                quote!(#(#stats)*)
            }
            Fields::Unnamed(fields) => {
                let fields = fields.unnamed.into_iter().collect::<Vec<_>>();
                let stats = map_compound(fields, &dst_crate_name);
                quote!(#(#stats)*)
            }
            _ => unimplemented!(),
        },
        _ => unimplemented!(),
    };

    let output = quote! {
        impl #generics #dst_crate_name ::H5TypeUnsized for #struct_name #generic_inputs {
            fn type_descriptor(&self) -> #dst_crate_name ::__internal::TypeDescriptor {
                #[allow(unused_mut)]
                let mut fields = vec![];
                let layout = ::core::alloc::Layout::new::<()>();
                #calculate_type
                let layout = layout.pad_to_align();
                debug_assert_eq!(layout, ::core::alloc::Layout::for_value(self));
                let ty = #dst_crate_name ::__internal::CompoundType { fields, size: layout.size() };
                #dst_crate_name ::__internal::TypeDescriptor::Compound(ty)
            }
        }
    };
    TokenStream::from(output)
}

fn map_compound(
    fields: impl IntoIterator<Item = Field>,
    dst_crate_name: &proc_macro2::TokenStream,
) -> Vec<proc_macro2::TokenStream> {
    fields
        .into_iter()
        .enumerate()
        .map(|(i, field)| {
            let name = field
                .ident
                .unwrap_or_else(|| parse_str::<Ident>(&i.to_string()).unwrap());
            let name_str = name.to_string();
            quote! {
                let new_layout = ::core::alloc::Layout::for_value(&self. #name);
                let (layout, offset) = layout.extend(new_layout).unwrap();
                fields.push(#dst_crate_name ::__internal::CompoundField::new(
                    #name_str,
                    #dst_crate_name ::H5TypeUnsized::type_descriptor(&self. #name),
                    offset,
                    #i,
                ));
            }
        })
        .collect()
}
