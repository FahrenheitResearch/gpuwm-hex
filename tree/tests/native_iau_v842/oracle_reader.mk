all: libiau_oracle_reader.so oracle_reader_tests

libiau_oracle_reader.so: oracle_reader.rs
	rustc --edition=2024 --crate-type=cdylib -O $< -o $@

oracle_reader_tests: oracle_reader.rs
	rustc --edition=2024 --test $< -o $@
