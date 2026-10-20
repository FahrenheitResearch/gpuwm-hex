all: analysis_increment analysis_increment_tests

analysis_increment: analysis_increment.rs
	rustc --edition=2024 -O $< -o $@

analysis_increment_tests: analysis_increment.rs
	rustc --edition=2024 --test $< -o $@
