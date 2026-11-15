"use client";
import { Widget } from "./widgets";
import { PART3, Widget3 } from "./widgets3";
import { PART2, Widget2 } from "./widgets2";
export default function WidgetRouter({ name }: { name: string }) {
  if (PART2.includes(name)) return <Widget2 name={name} />;
  return PART3.includes(name) ? <Widget3 name={name} /> : <Widget name={name} />;
}
