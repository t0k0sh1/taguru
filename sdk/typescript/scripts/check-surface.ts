#!/usr/bin/env npx tsx
/**
 * Check the TypeScript SDK against sdk/spec/surface.yaml.
 *
 * Method names are derived from the spec's canonical Python names by
 * snake_case → camelCase. Verifies every declared method exists with exactly
 * the declared positional parameters and trailing-options-object keys, and no
 * undeclared public method exists. Run from sdk/typescript (needs its
 * node_modules): `npm run check:surface` from sdk/typescript.
 */

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import ts from "typescript";
import { parse } from "yaml";

const here = dirname(fileURLToPath(import.meta.url));
const SPEC_PATH = resolve(here, "../../spec/surface.yaml");
const CLIENT_PATH = resolve(here, "../src/client.ts");
const MODELS_PATH = resolve(here, "../src/models.ts");

interface MethodSpec {
  route?: string;
  args?: string[];
  options?: string[];
}

interface Spec {
  functions?: Record<string, MethodSpec>;
  classes?: Record<string, Record<string, MethodSpec | null>>;
}

const camel = (snake: string): string => snake.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());

interface ParsedMethod {
  args: string[];
  options: string[];
}

function parseParameters(parameters: readonly ts.ParameterDeclaration[]): ParsedMethod {
  const args: string[] = [];
  const options: string[] = [];
  for (const parameter of parameters) {
    const name = parameter.name.getText();
    const type = parameter.type;
    if (name === "options" && type !== undefined && ts.isTypeLiteralNode(type)) {
      for (const member of type.members) {
        if (ts.isPropertySignature(member) && member.name !== undefined) {
          options.push(member.name.getText());
        }
      }
      continue;
    }
    args.push(name);
  }
  return { args, options };
}

function isPublicApi(member: ts.MethodDeclaration): boolean {
  const modifiers = ts.getModifiers(member) ?? [];
  if (modifiers.some((mod) => mod.kind === ts.SyntaxKind.PrivateKeyword)) {
    return false;
  }
  const tags = ts.getJSDocTags(member);
  return !tags.some((tag) => tag.tagName.text === "internal");
}

function collectClasses(source: ts.SourceFile): Map<string, Map<string, ParsedMethod>> {
  const classes = new Map<string, Map<string, ParsedMethod>>();
  source.forEachChild((node) => {
    if (!ts.isClassDeclaration(node) || node.name === undefined) {
      return;
    }
    const methods = new Map<string, ParsedMethod>();
    for (const member of node.members) {
      if (ts.isMethodDeclaration(member) && ts.isIdentifier(member.name) && isPublicApi(member)) {
        methods.set(member.name.text, parseParameters(member.parameters));
      }
    }
    classes.set(node.name.text, methods);
  });
  return classes;
}

function collectFunctions(source: ts.SourceFile): Map<string, ParsedMethod> {
  const functions = new Map<string, ParsedMethod>();
  source.forEachChild((node) => {
    if (ts.isFunctionDeclaration(node) && node.name !== undefined) {
      functions.set(node.name.text, parseParameters(node.parameters));
    }
  });
  return functions;
}

function main(): number {
  const spec = parse(readFileSync(SPEC_PATH, "utf-8")) as Spec;
  const errors: string[] = [];

  const parseSource = (path: string): ts.SourceFile =>
    ts.createSourceFile(path, readFileSync(path, "utf-8"), ts.ScriptTarget.ES2022, true);

  const classes = collectClasses(parseSource(CLIENT_PATH));
  const functions = collectFunctions(parseSource(MODELS_PATH));

  for (const [funcName, declared] of Object.entries(spec.functions ?? {})) {
    const tsName = camel(funcName);
    const found = functions.get(tsName);
    if (found === undefined) {
      errors.push(`missing exported function: ${tsName}`);
      continue;
    }
    const wantArgs = declared.args ?? [];
    if (JSON.stringify(found.args) !== JSON.stringify(wantArgs)) {
      errors.push(`${tsName}: args [${found.args}] != spec [${wantArgs}]`);
    }
  }

  for (const [className, methods] of Object.entries(spec.classes ?? {})) {
    const actual = classes.get(className);
    if (actual === undefined) {
      errors.push(`missing class: ${className}`);
      continue;
    }
    const declaredNames = new Set(Object.keys(methods).map(camel));
    for (const declared of [...declaredNames].sort()) {
      if (!actual.has(declared)) {
        errors.push(`${className}.${declared}: declared in spec but not implemented`);
      }
    }
    for (const extra of [...actual.keys()].sort()) {
      if (!declaredNames.has(extra)) {
        errors.push(`${className}.${extra}: public method not declared in spec`);
      }
    }
    for (const [pyName, entry] of Object.entries(methods)) {
      const tsName = camel(pyName);
      const found = actual.get(tsName);
      if (found === undefined) {
        continue;
      }
      const wantArgs = entry?.args ?? [];
      const wantOptions = entry?.options ?? [];
      if (JSON.stringify(found.args) !== JSON.stringify(wantArgs)) {
        errors.push(`${className}.${tsName}: args [${found.args}] != spec [${wantArgs}]`);
      }
      if (JSON.stringify(found.options) !== JSON.stringify(wantOptions)) {
        errors.push(`${className}.${tsName}: options [${found.options}] != spec [${wantOptions}]`);
      }
    }
  }

  if (errors.length > 0) {
    console.error(`surface parity check FAILED (${errors.length} problems):`);
    for (const error of errors) {
      console.error(`  - ${error}`);
    }
    return 1;
  }
  console.log("typescript surface matches sdk/spec/surface.yaml");
  return 0;
}

process.exit(main());
