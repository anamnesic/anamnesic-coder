use anyhow::Context;

/// Fetch a URL (http/https) and return its text content. HTML pages are
/// stripped to readable text. The response body is capped at `max_bytes`.
pub fn http_fetch(url: &str, max_bytes: usize, timeout_secs: u64) -> anyhow::Result<String> {
    let parsed = reqwest::Url::parse(url).context("invalid URL")?;
    if !matches!(parsed.scheme(), "http" | "https") {
        anyhow::bail!("only http/https URLs are allowed");
    }
    let client = reqwest::blocking::Client::builder()
        .user_agent(web_ua())
        .timeout(std::time::Duration::from_secs(timeout_secs.max(1)))
        .build()
        .context("building HTTP client")?;
    let response = client.get(parsed).send().context("HTTP request failed")?;
    if !response.status().is_success() {
        anyhow::bail!("HTTP {}", response.status());
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body = response.bytes().context("reading response body")?;
    let body = String::from_utf8_lossy(&body).into_owned();
    let text = if content_type.contains("html") {
        strip_html(&body)
    } else {
        body
    };
    Ok(truncate_chars(&text, max_bytes))
}

/// Web search with no API key. Uses a SearXNG instance when `WEB_SEARCH_URL`
/// is set (JSON endpoint), otherwise falls back to DuckDuckGo's HTML results.
pub fn web_search(query: &str, max_results: usize, timeout_secs: u64) -> anyhow::Result<String> {
    let query = query.trim();
    if query.is_empty() {
        anyhow::bail!("query must not be empty");
    }
    let client = reqwest::blocking::Client::builder()
        .user_agent(web_ua())
        .timeout(std::time::Duration::from_secs(timeout_secs.max(1)))
        .build()
        .context("building HTTP client")?;

    match std::env::var("WEB_SEARCH_URL").ok() {
        Some(searxng_url) => searxng_search(&client, &searxng_url, query, max_results),
        None => duckduckgo_search(&client, query, max_results),
    }
}

fn searxng_search(
    client: &reqwest::blocking::Client,
    searxng_url: &str,
    query: &str,
    max_results: usize,
) -> anyhow::Result<String> {
    let url = reqwest::Url::parse_with_params(searxng_url, &[("q", query), ("format", "json")])
        .context("invalid WEB_SEARCH_URL")?;
    let response = client.get(url).send().context("SearXNG request failed")?;
    if !response.status().is_success() {
        anyhow::bail!("SearXNG HTTP {}", response.status());
    }
    let json: serde_json::Value =
        response.json().context("SearXNG returned invalid JSON")?;
    let results = json
        .get("results")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for result in results.iter().take(max_results.clamp(1, 20)) {
        let title = result
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("(untitled)");
        let url = result
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let snippet = result
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        out.push(format!("{title}\n{url}\n{snippet}"));
    }
    if out.is_empty() {
        anyhow::bail!("no results for query");
    }
    Ok(out.join("\n\n"))
}

fn duckduckgo_search(
    client: &reqwest::blocking::Client,
    query: &str,
    max_results: usize,
) -> anyhow::Result<String> {
    let url = reqwest::Url::parse_with_params(
        "https://html.duckduckgo.com/html/",
        &[("q", query)],
    )?;
    let response = client.get(url).send().context("DuckDuckGo request failed")?;
    if !response.status().is_success() {
        anyhow::bail!("DuckDuckGo HTTP {}", response.status());
    }
    let body = response.text().context("reading DuckDuckGo response")?;
    let results = parse_ddg_results(&body, max_results.clamp(1, 20));
    if results.is_empty() {
        anyhow::bail!("no results for query");
    }
    Ok(results.join("\n\n"))
}

/// Extract DuckDuckGo HTML results (`result__a` title links + `result__snippet`
/// summaries) with lightweight regex parsing — no HTML dependency needed.
fn parse_ddg_results(html: &str, max_results: usize) -> Vec<String> {
    let anchor_re = regex::Regex::new(
        r#"class="[^"]*result__a[^"]*" href="([^"]+)"[^>]*>(.*?)</a>"#,
    )
    .expect("valid anchor regex");
    let snippet_re =
        regex::Regex::new(r#"class="result__snippet"[^>]*>(.*?)</a>"#).expect("valid snippet regex");
    let strip = regex::Regex::new(r"<[^>]+>").expect("valid strip regex");
    let anchors: Vec<(String, String)> = anchor_re
        .captures_iter(html)
        .map(|cap| {
            let href = html_unescape(&cap[1]);
            let title = strip.replace_all(&html_unescape(&cap[2]), "").trim().to_string();
            (href, title)
        })
        .collect();
    let snippets: Vec<String> = snippet_re
        .captures_iter(html)
        .map(|cap| strip.replace_all(&html_unescape(&cap[1]), "").trim().to_string())
        .collect();
    anchors
        .into_iter()
        .take(max_results)
        .enumerate()
        .map(|(index, (href, title))| {
            let snippet = snippets.get(index).cloned().unwrap_or_default();
            format!("{title}\n{href}\n{snippet}")
        })
        .collect()
}

fn html_unescape(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
}

fn strip_html(html: &str) -> String {
    let strip_tags = regex::Regex::new(r"<[^>]+>").expect("valid tag regex");
    let strip_scripts =
        regex::Regex::new(r"(?is)<script.*?</script>|<style.*?</style>").expect("valid block regex");
    let whitespace = regex::Regex::new(r"[ \t\r\f\v]{2,}").expect("valid whitespace regex");
    let no_blocks = strip_scripts.replace_all(html, " ");
    let text = strip_tags.replace_all(&no_blocks, " ");
    let text = whitespace.replace_all(&text, " ");
    text.trim().to_string()
}

fn truncate_chars(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…[truncated]", &text[..end])
}

fn web_ua() -> String {
    format!("anamnesic-coder/{}", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_html_removes_tags() {
        let html = "<html><body><h1>Title</h1><p>Hello <b>world</b>.</p></body></html>";
        assert_eq!(strip_html(html), "Title Hello world .");
    }

    #[test]
    fn truncate_keeps_utf8_boundaries() {
        let text = "açãodólar".repeat(1000);
        let capped = truncate_chars(&text, 10);
        assert!(capped.len() <= 10 + "[truncated]".len() + 8);
        assert!(String::from_utf8(capped.into_bytes()).is_ok());
    }

    #[test]
    fn ddg_parser_extracts_results() {
        let html = r#"
            <a class="result__a" href="//example.com/1">First <b>Result</b></a>
            <a class="result__snippet" href="//example.com/1">A snippet.</a>
            <a class="result__a" href="//example.com/2">Second</a>
            <a class="result__snippet" href="//example.com/2">More text.</a>
        "#;
        let results = parse_ddg_results(html, 2);
        assert_eq!(results.len(), 2);
        assert!(results[0].contains("First Result"));
        assert!(results[0].contains("//example.com/1"));
        assert!(results[0].contains("A snippet."));
    }

    #[test]
    #[ignore = "live network"]
    fn http_fetch_live_strips_html() {
        let text = http_fetch("https://example.com/", 8_000, 20).unwrap();
        assert!(text.contains("Example Domain"), "got: {text}");
        assert!(!text.contains("<html"), "HTML not stripped: {text}");
    }

    #[test]
    #[ignore = "live network"]
    fn web_search_live_returns_results() {
        let results = web_search("rust programming language", 3, 20).unwrap();
        assert!(results.contains("rust"), "got: {results}");
    }
}
