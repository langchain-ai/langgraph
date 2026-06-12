// Package lgexec provides subprocess execution helpers for the LangGraph CLI.
//
// The package name is lgexec (rather than exec) to avoid shadowing the
// standard library os/exec package.
package lgexec

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
)

// RunOpts configures how a subprocess is executed.
type RunOpts struct {
	Stdin   string   // input to pass via stdin
	Verbose bool     // pipe stdout/stderr to os.Stdout/os.Stderr
	Dir     string   // working directory
	Env     []string // environment variables (KEY=VALUE)
}

// Run executes the named program with the given arguments.
//
// When Verbose is true stdout and stderr are forwarded to the process's
// os.Stdout / os.Stderr.  Otherwise output is silently discarded.
// A non-zero exit code is returned as an *ExitError.
func Run(name string, args []string, opts RunOpts) error {
	cmd := exec.Command(name, args...)

	if opts.Dir != "" {
		cmd.Dir = opts.Dir
	}
	if len(opts.Env) > 0 {
		cmd.Env = append(os.Environ(), opts.Env...)
	}

	if opts.Stdin != "" {
		cmd.Stdin = strings.NewReader(opts.Stdin)
	}

	if opts.Verbose {
		if opts.Stdin != "" {
			cmdStr := fmt.Sprintf("+ %s %s", name, strings.Join(args, " "))
			fmt.Printf("%s <\n%s\n", cmdStr, strings.Join(
				nonEmptyLines(opts.Stdin), "\n"))
		} else {
			fmt.Printf("+ %s %s\n", name, strings.Join(args, " "))
		}
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	} else {
		cmd.Stdout = io.Discard
		cmd.Stderr = io.Discard
	}

	return cmd.Run()
}

// RunCollect executes the named program and collects stdout and stderr.
// Both are returned as strings. A non-zero exit code results in a non-nil error.
func RunCollect(name string, args []string) (stdout, stderr string, err error) {
	cmd := exec.Command(name, args...)

	var outBuf, errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf

	err = cmd.Run()
	return outBuf.String(), errBuf.String(), err
}

// RunWithCallback executes the named program and invokes onStdout for each
// line of stdout output.  If onStdout returns true the callback is no longer
// called and remaining stdout is forwarded directly to os.Stdout (matching
// the Python CLI's monitor_stream behaviour).
// Stderr is always forwarded to os.Stderr.
func RunWithCallback(name string, args []string, onStdout func(string) bool) error {
	cmd := exec.Command(name, args...)
	cmd.Stderr = os.Stderr

	pipe, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("cannot create stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("cannot start command: %w", err)
	}

	scanner := bufio.NewScanner(pipe)
	stopped := false
	for scanner.Scan() {
		line := scanner.Text()
		if stopped {
			// After callback signalled stop, forward remaining output.
			fmt.Fprintln(os.Stdout, line)
			continue
		}
		if onStdout(line) {
			stopped = true
		}
	}
	if scanErr := scanner.Err(); scanErr != nil {
		// Drain but ignore read errors on stdout — the exit code matters.
		_ = scanErr
	}

	return cmd.Wait()
}

// nonEmptyLines splits s on newlines and returns lines that are not empty.
func nonEmptyLines(s string) []string {
	var out []string
	for _, line := range strings.Split(s, "\n") {
		if line != "" {
			out = append(out, line)
		}
	}
	return out
}
