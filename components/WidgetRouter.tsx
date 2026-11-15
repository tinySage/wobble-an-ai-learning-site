"use client";
import { Widget } from "./widgets";
import { PART3, Widget3 } from "./widgets3";
import { PART2, Widget2 } from "./widgets2";
import { PART4, Widget4 } from "./widgets4";
import { PART5, Widget5 } from "./widgets5";
export default function WidgetRouter({ name }: { name: string }) {
  if (PART5.includes(name)) return <Widget5 name={name} />;
  if (PART4.includes(name)) return <Widget4 name={name} />;
  if (PART2.includes(name)) return <Widget2 name={name} />;
  return PART3.includes(name) ? <Widget3 name={name} /> : <Widget name={name} />;
}
