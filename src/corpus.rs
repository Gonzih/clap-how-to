use clap::Command;
use serde::{Deserialize, Serialize};

/// Original, caller-owned location of a result. No URL or command is executed.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Source {
    /// Canonical command path and its exact help invocation.
    Command {
        /// Program and subcommand names, without invented arguments.
        command: String,
        /// The command followed by `--help`.
        help: String,
    },
    /// An exact excerpt from a caller-supplied Markdown source.
    Documentation {
        /// Source filename or caller-defined document identifier.
        source: String,
        /// Heading containing the excerpt.
        section: String,
        /// One-based line of its first source line.
        line: usize,
        /// Caller-supplied full-document URL; never fetched.
        #[serde(skip_serializing_if = "Option::is_none")]
        url: Option<String>,
    },
    /// Historical caller-owned record, such as a local harness log entry.
    Record {
        /// Caller-defined provider or record family.
        provider: String,
        /// File or other caller-owned locator; never opened by this library.
        source: String,
        /// Byte offset of the original record in that source.
        byte_offset: u64,
    },
}

/// A command description or real documentation excerpt to index.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Document {
    /// Short human-readable title, also used for retrieval.
    pub title: String,
    /// Source text, never a generated answer.
    pub text: String,
    /// Provenance retained in every hit.
    #[serde(flatten)]
    pub source: Source,
}

impl Document {
    pub(crate) fn group(&self) -> String {
        match &self.source {
            Source::Command { command, .. } => format!("command:{command}"),
            Source::Documentation {
                source, section, ..
            } => format!("doc:{source}#{section}"),
            Source::Record {
                provider,
                source,
                byte_offset,
            } => format!("record:{provider}:{source}:{byte_offset}"),
        }
    }
    pub(crate) fn indexed_text(&self) -> String {
        format!("{} {} {}", self.title, self.title, self.text)
    }
    pub(crate) fn semantic_text(&self) -> String {
        // Intent descriptions carry meaning; repeated titles and generated
        // argument boilerplate can dominate small sentence encoders. Keep the
        // complete text in BM25 and the displayed result.
        let text = match self.source {
            Source::Command { .. } => self.text.lines().next().unwrap_or(&self.text),
            Source::Documentation { .. } | Source::Record { .. } => &self.text,
        };
        format!("{}: {text}", self.title)
    }
}

/// Extract canonical visible commands, aliases, local arguments, possible values
/// and after-help text (long form preferred).
///
/// Operates on a clone; no parsing, completion handler, program execution or I/O.
/// Hidden commands/arguments and generated help/discovery commands are omitted.
pub fn command_documents(command: &Command) -> Vec<Document> {
    fn visit(command: &Command, program: &str, path: &str, out: &mut Vec<Document>) {
        if command.is_hide_set() || matches!(command.get_name(), "help" | "how" | "how-to") {
            return;
        }
        let about = command
            .get_long_about()
            .or(command.get_about())
            .map(ToString::to_string)
            .unwrap_or_default();
        let arguments = command
            .get_arguments()
            .filter(|arg| {
                !arg.is_hide_set()
                    && (path.is_empty() || !arg.is_global_set())
                    && !matches!(arg.get_id().as_str(), "help" | "version")
            })
            .map(|arg| {
                format!(
                    "{} {} {}",
                    arg.get_long().unwrap_or(arg.get_id().as_str()),
                    arg.get_long_help()
                        .or(arg.get_help())
                        .map(ToString::to_string)
                        .unwrap_or_default(),
                    arg.get_value_parser()
                        .possible_values()
                        .map(|values| values
                            .map(|v| v.get_name().to_owned())
                            .collect::<Vec<_>>()
                            .join(" "))
                        .unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let canonical = if path.is_empty() {
            program.into()
        } else {
            format!("{program} {path}")
        };
        let aliases = command.get_visible_aliases().collect::<Vec<_>>().join(" ");
        let text = format!("{about}\n{arguments} {aliases}").trim().to_owned();
        // Establish the existing intent/fallback before appending lexical-only
        // details, including for commands with no about text or arguments.
        let mut text = if text.is_empty() {
            canonical.clone()
        } else {
            text
        };
        if let Some(details) = command.get_after_long_help().or(command.get_after_help()) {
            text.push('\n');
            text.push_str(&details.to_string());
        }
        out.push(Document {
            title: if path.is_empty() {
                program.into()
            } else {
                path.into()
            },
            text,
            source: Source::Command {
                help: format!("{canonical} --help"),
                command: canonical,
            },
        });
        for child in command.get_subcommands() {
            let child_path = if path.is_empty() {
                child.get_name().into()
            } else {
                format!("{path} {}", child.get_name())
            };
            visit(child, program, &child_path, out);
        }
    }
    let mut command = command.clone();
    command.build();
    let mut documents = Vec::new();
    visit(&command, command.get_name(), "", &mut documents);
    documents
}

/// Chunk Markdown into exact source excerpts with heading/line provenance.
/// Chunks normally have at most 700 bytes; an introduction and its example may
/// be retained together up to 1400 bytes. YAML frontmatter is not source guidance.
///
/// Fenced headings remain code. Search-query examples are excluded to avoid
/// matching the question instead of its answer. Search deduplicates a section.
pub fn markdown_documents(source: &str, markdown: &str, url: Option<&str>) -> Vec<Document> {
    fn block(
        source: &str,
        section: &str,
        mut line: usize,
        lines: &[&str],
        url: Option<&str>,
        out: &mut Vec<Document>,
    ) {
        let text = lines.join("\n");
        if [" cli search ", " docs search ", " how-to \"", " how \""]
            .iter()
            .any(|phrase| text.contains(phrase))
        {
            return;
        }
        let mut rest = text.as_str();
        while !rest.is_empty() {
            let trimmed = rest.trim_start();
            line += rest[..rest.len() - trimmed.len()]
                .bytes()
                .filter(|&b| b == b'\n')
                .count();
            rest = trimmed;
            if rest.is_empty() {
                break;
            }
            let end = if rest.len() <= 700 {
                rest.len()
            } else {
                // Balance the final chunks rather than leaving an orphaned
                // sentence fragment (which can outrank real context).
                let target = rest.len().div_ceil(rest.len().div_ceil(700));
                let end = rest
                    .char_indices()
                    .map(|(i, _)| i)
                    .take_while(|&i| i <= target)
                    .last()
                    .unwrap();
                rest[..end]
                    .rfind(char::is_whitespace)
                    .filter(|&i| i > target / 2)
                    .unwrap_or(end)
            };
            let excerpt = rest[..end].trim();
            if !excerpt.is_empty() {
                out.push(Document {
                    title: section.into(),
                    text: excerpt.into(),
                    source: Source::Documentation {
                        source: source.into(),
                        section: section.into(),
                        line,
                        url: url.map(str::to_owned),
                    },
                });
            }
            line += rest[..end].bytes().filter(|&b| b == b'\n').count();
            rest = &rest[end..];
        }
    }
    let mut result = Vec::new();
    let frontmatter_end = markdown
        .lines()
        .next()
        .filter(|line| *line == "---")
        .and_then(|_| {
            markdown
                .lines()
                .enumerate()
                .skip(1)
                .find(|(_, line)| *line == "---")
                .map(|(i, _)| i)
        });
    let (mut section, mut start, mut fenced) = (source.to_owned(), 1, false);
    let mut lines = Vec::new();
    for (i, line) in markdown.lines().enumerate() {
        if frontmatter_end.is_some_and(|end| i <= end) {
            continue;
        }
        let heading = (!fenced)
            .then(|| {
                line.strip_prefix('#')
                    .map(|s| s.trim_start_matches('#'))
                    .and_then(|s| s.strip_prefix(' '))
            })
            .flatten();
        if let Some(title) = heading {
            block(source, &section, start, &lines, url, &mut result);
            lines.clear();
            section = title.trim().into();
            continue;
        }
        if line.is_empty() && !fenced {
            block(source, &section, start, &lines, url, &mut result);
            lines.clear();
            continue;
        }
        if lines.is_empty() {
            start = i + 1;
        }
        lines.push(line);
        if line.starts_with("```") {
            fenced = !fenced;
        }
    }
    block(source, &section, start, &lines, url, &mut result);
    complete_excerpts(result, markdown)
}

fn introduction(text: &str) -> bool {
    text.trim_end_matches(&['*', '_'][..]).ends_with(':')
}

/// Keep a short introduction with its example, preserving exact source lines.
/// Standalone fragments such as "Read messages with:" cannot answer a question.
fn complete_excerpts(mut documents: Vec<Document>, markdown: &str) -> Vec<Document> {
    let lines: Vec<_> = markdown.lines().collect();
    let mut index = 0;
    while index + 1 < documents.len() {
        let current = &documents[index];
        let next = &documents[index + 1];
        if let (
            Source::Documentation { section, line, .. },
            Source::Documentation {
                section: next_section,
                line: next_line,
                ..
            },
        ) = (&current.source, &next.source)
        {
            let start = line - 1;
            let end = start + current.text.lines().count();
            let next_start = next_line - 1;
            let next_end = next_start + next.text.lines().count();
            if introduction(&current.text)
                && section == next_section
                && end <= next_start
                && lines[start..end].join("\n") == current.text
                && lines[next_start..next_end].join("\n") == next.text
                && lines[end..next_start]
                    .iter()
                    .all(|line| line.trim().is_empty())
                && lines[start..next_end]
                    .iter()
                    .map(|line| line.len() + 1)
                    .sum::<usize>()
                    <= 1400
            {
                documents[index].text = lines[start..next_end].join("\n");
                documents.remove(index + 1);
                continue;
            }
        }
        index += 1;
    }
    documents.retain(|document| !introduction(&document.text));
    documents
}
