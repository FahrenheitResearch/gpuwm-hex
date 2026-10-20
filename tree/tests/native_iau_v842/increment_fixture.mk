all: increment_fixture

increment_fixture: increment_fixture.rs
	rustc --edition=2024 -O $< -o $@
