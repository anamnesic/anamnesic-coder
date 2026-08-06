use crate::repo::{SymbolEntry, SymbolIndex};
use std::path::Path;

pub fn search_symbols(workspace: &Path, query: &str, limit: usize) -> Vec<SymbolEntry> {
    let index = SymbolIndex::build(workspace);
    index
        .search(query, limit)
        .into_iter()
        .cloned()
        .collect()
}

pub fn search_symbols_by_type(workspace: &Path, symbol_type: &str, limit: usize) -> Vec<SymbolEntry> {
    let index = SymbolIndex::build(workspace);
    index
        .search_type(symbol_type, limit)
        .into_iter()
        .cloned()
        .collect()
}

pub fn format_symbols(symbols: &[SymbolEntry]) -> String {
    if symbols.is_empty() {
        return "No matching symbols found.".to_string();
    }
    let mut lines = Vec::new();
    let mut current_file = String::new();
    for sym in symbols {
        if sym.file_path != current_file {
            current_file = sym.file_path.clone();
            lines.push(format!("\nFile `{}`:", current_file));
        }
        lines.push(format!(
            "  L{:<4} {} {}",
            sym.line_number, sym.symbol_type, sym.name
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn searches_rust_symbols() {
        let temp = std::env::temp_dir().join(format!("test_symbol_search_{}", std::process::id()));
        fs::create_dir_all(temp.join("src")).unwrap();
        fs::write(
            temp.join("src/lib.rs"),
            "pub fn parse_input() {}\npub struct Data {}\nfn internal() {}\n",
        )
        .unwrap();

        let results = search_symbols(&temp, "parse", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "parse_input");
        assert_eq!(results[0].symbol_type, "fn");

        let results = search_symbols(&temp, "Data", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].symbol_type, "type");

        let results = search_symbols(&temp, "internal", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].symbol_type, "fn");

        fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn searches_by_type() {
        let temp = std::env::temp_dir().join(format!("test_symbol_search_type_{}", std::process::id()));
        fs::create_dir_all(temp.join("src")).unwrap();
        fs::write(
            temp.join("src/lib.rs"),
            "pub fn foo() {}\npub struct Bar {}\npub fn baz() {}\n",
        )
        .unwrap();

        let fns = search_symbols_by_type(&temp, "fn", 10);
        assert_eq!(fns.len(), 2);
        assert!(fns.iter().all(|s| s.symbol_type == "fn"));

        let types = search_symbols_by_type(&temp, "type", 10);
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].name, "struct Bar");

        fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn formats_symbols() {
        let symbols = vec![
            SymbolEntry {
                file_path: "src/lib.rs".into(),
                symbol_type: "fn",
                name: "foo".into(),
                line_number: 1,
            },
            SymbolEntry {
                file_path: "src/lib.rs".into(),
                symbol_type: "type",
                name: "struct Bar".into(),
                line_number: 3,
            },
        ];
        let formatted = format_symbols(&symbols);
        assert!(formatted.contains("File `src/lib.rs`:"));
        assert!(formatted.contains("fn foo"));
        assert!(formatted.contains("type struct Bar"));
    }
}